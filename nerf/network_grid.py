import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp, biased_softplus
from .renderer import NeRFRenderer
import raymarching
import numpy as np
from encoding import get_encoder
from .volumetric_rendering.renderer import generate_planes, sample_from_planes
from .volumetric_rendering import math_utils
from. volumetric_rendering.ray_marcher import MipRayMarcher2

from .utils import safe_normalize

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=3,
                 hidden_dim=64,
                 num_layers_bg=2,
                 hidden_dim_bg=32,
                 ):
        
        super().__init__(opt)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # self.encoder, self.in_dim = get_encoder('hashgrid', input_dim=3, log2_hashmap_size=19, desired_resolution=2048 * self.bound, interpolation='smoothstep')

        # self.sigma_net = MLP(self.in_dim, 4, hidden_dim, num_layers, bias=True)
        self.decoder = OSGDecoder(32, 3).float()
        self.planes = triplane()
        self.plane_axes = generate_planes()
        # self.normal_net = MLP(self.in_dim, 3, hidden_dim, num_layers, bias=True)

        self.density_activation = trunc_exp if self.opt.density_activation == 'exp' else biased_softplus

        # background network
        if self.opt.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            
            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3, multires=6)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None

    def common_forward(self, x):
        print(f"common_forward x.shape {x.shape}")
        self.plane_axes = self.plane_axes.to(x.device)
        sampled_features = sample_from_planes(self.plane_axes, self.planes(), x.unsqueeze(0), padding_mode='zeros', box_warp=2*self.bound)
        out = self.decoder(sampled_features)
        # print(f"out['sigma'] {out['sigma'].grad.sum()}")
        return out['sigma'].reshape(-1,1).float(), out['rgb'].reshape(-1,3).float()
        # sigma
        # print(f"x.dtype {x.dtype}")
        # enc = self.encoder(x, bound=self.bound, max_level=self.max_level)

        # print(f"enc.dtype {enc.dtype}")
        # h = self.sigma_net(enc)

        # print(f"h.dtype {h.dtype}")
        # sigma = self.density_activation(h[..., 0] + self.density_blob(x))
        # albedo = torch.sigmoid(h[..., 1:])

        # return sigma, albedo
    
    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal

    def normal(self, x):
        normal = self.finite_difference_normal(x)
        normal = safe_normalize(normal)
        normal = torch.nan_to_num(normal)
        return normal
    
    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        sigma, albedo = self.common_forward(x)

        if shading == 'albedo':
            normal = None
            color = albedo
        
        else: # lambertian shading

            # normal = self.normal_net(enc)
            normal = self.normal(x)

            
            if shading == 'textureless':
                lambertian = ratio + (1 - ratio) * (normal * l).sum(-1).clamp(min=0) # [N,]
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else: # 'lambertian'
                lambertian = ratio + (1 - ratio) * (normal * l).sum(-1).clamp(min=0) # [N,]
                color = albedo * lambertian.unsqueeze(-1)
            
        return sigma, color, normal

      
    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        
        sigma, albedo = self.common_forward(x)
        
        return {
            'sigma': sigma,
            'albedo': albedo,
        }


    def background(self, d):

        h = self.encoder_bg(d) # [N, C]
        
        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.planes.parameters(), 'lr': lr * 10},
            {'params': self.decoder.parameters(), 'lr': lr},
            # {'params': self.normal_net.parameters(), 'lr': lr},
        ]        

        if self.opt.bg_radius > 0:
            # params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        if self.opt.dmtet and not self.opt.lock_geo:
            params.append({'params': self.sdf, 'lr': lr})
            params.append({'params': self.deform, 'lr': lr})

        return params



class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, decoder_output_dim):
        super().__init__()
        self.hidden_dim = 64
        self.decoder_output_dim = decoder_output_dim

        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            torch.nn.Softplus(),
            nn.Linear(self.hidden_dim, 1 + decoder_output_dim)
        )
        
    def forward(self, sampled_features):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)
        
        x = self.net(x)
        x = x.view(N, M, 4)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}

class triplane(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.planes = nn.Parameter(torch.ones(1,96,128,128))
        self.net = nn.Sequential(*([nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1),\
                                  nn.BatchNorm2d(num_features=96),\
                                  nn.ReLU()]*2))
    def forward(self,):
        return self.net(self.planes).reshape(1,3,32,128,128)
