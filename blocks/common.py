import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


class BaseModule(nn.Module):
    def _init_weights(self, m: nn.Module):
        for name, p in m.named_parameters():
            if name in ["out_proj.weight"]:
                p = p.clone().detach_() # fake init, just to keep the seed ....
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return rearrange(x, 'b h w c -> b c h w')
