import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from blocks.common import BaseModule
from blocks.patch import PatchEmbed2D, PatchMerging2D, PatchExpand2D, Final_PatchExpand2D
from blocks.ssm import VSSBlock
from blocks.vit import IBIBlock
from blocks.cnn import CBAM



class VSSModule(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        init_value: float = 1.0,
        vss_block: Callable[..., torch.nn.Module] = VSSBlock,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vssm = vss_block(hidden_dim, drop_path, norm_layer, attn_drop_rate, d_state, **kwargs)
        self.conv_bbone = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim),
                                        nn.BatchNorm2d(num_features=hidden_dim),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                                        nn.BatchNorm2d(num_features=hidden_dim),
                                        nn.SiLU())
        if not init_value == None:
            self.lambda_ = nn.Parameter(init_value*torch.ones(hidden_dim,1,1), requires_grad=True)
            self.beta =  nn.Parameter(init_value*torch.ones(hidden_dim,1,1), requires_grad=True)
        self.mlp = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                                 nn.BatchNorm2d(num_features=hidden_dim),
                                 nn.SiLU())
        def forward(self, x):
            raise NotImplementedError


class VSSModuleV1(VSSModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        y_ssm = self.lambda_ * self.vssm(x)
        y_cnn = self.conv_bbone(y_ssm + x)
        x = y_cnn + self.beta*x
        x = self.mlp(x)
        return x
    

class VSSModuleV2(VSSModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x_vss = self.lambda_ * self.vssm(x)
        x_cnn = self.beta * self.conv_bbone(x_vss + x)
        x = self.mlp(x_vss + x_cnn)
        return x


class VSSModuleV3(VSSModule):
    def __init__(self, *args, **kwargs):
        kwargs['init_value'] = None
        fmap_size = kwargs['fmap_size']        
        
        super().__init__(*args, **kwargs)
        stage_spec = kwargs.get('stage_spec', 'L')
        depths = len(stage_spec)
        
        self.cbbam = CBAM(self.hidden_dim, ratio=16, kernel_size=5)
        self.ibi = IBIBlock(fmap_size, 
                            window_size=7, 
                            dim_in=self.hidden_dim, 
                            dim_embed=self.hidden_dim, 
                            depths=depths, stage_spec=stage_spec, heads=4, 
                            attn_drop=0.0, proj_drop=0.0, expansion_mlp=1,
                            drop=0.0, drop_path_rate=0.0, use_dwc_mlp=False)

    def forward(self, x):
        x_cbb = self.conv_bbone(x)
        x_cnn = self.cbbam(x_cbb) + x
        x_ibi, _, _ = self.ibi(x_cnn)
        x = x_ibi + x_cnn
        x_vss = self.vssm(x)
        x = self.mlp(x_vss + x)
        return x


class VSSLayer(BaseModule):
    """ A layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        upsample=None,
        downsample=None,
        use_checkpoint=False,
        d_state=16,
        init_value: float =1.0,
        vss_module: Callable[..., torch.nn.Module] = partial(VSSModuleV1, vss_block=VSSBlock),
        spatial_dim=224,
        window_size=7,
        **kwargs,
    ):
        stage_specs = kwargs.get('stage_spec', "L")
        fmap_size = spatial_dim // window_size
        
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([vss_module(
            hidden_dim = dim,
            drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer = norm_layer,
            attn_drop_rate = attn_drop,
            d_state = d_state,
            init_value = init_value,
            fmap_size = fmap_size,
            window_size = window_size,
            stage_spec = stage_specs,
        ) for i in range(depth)])

        self.upsample = upsample(dim, norm_layer=norm_layer) if callable(upsample) else None
        self.downsample = downsample(dim, norm_layer=norm_layer) if callable(downsample) else None
        self.apply(self._init_weights)        

    def forward(self, x):
        if self.upsample: x = self.upsample(x)
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        if self.downsample: x = self.downsample(x)
        return x

VSSLayerV1 = partial(VSSLayer, vss_module=VSSModuleV1)
VSSLayerV2 = partial(VSSLayer, vss_module=VSSModuleV2)
VSSLayerV3 = partial(VSSLayer, vss_module=VSSModuleV3)

VSSLayer_down_V1 = partial(VSSLayerV1, upsample=None)
VSSLayer_down_V2 = partial(VSSLayerV2, upsample=None)
VSSLayer_down_V3 = partial(VSSLayerV3, upsample=None)
VSSLayer_up_V1 = partial(VSSLayerV1, downsample=None)
VSSLayer_up_V2 = partial(VSSLayerV2, downsample=None)
VSSLayer_up_V3 = partial(VSSLayerV3, downsample=None)


class VMACO(BaseModule):
    def __init__(self, patch_size=4, in_chans=3, num_classes=9, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, 
                 vss_layer=VSSLayerV1,
                 spatial_size=224, stage_specs=["L", "L", "L", "L"], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.patch_embed = PatchEmbed2D(patch_size, in_chans, embed_dim=self.embed_dim, norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        fmas = [56, 28, 14, 7]
        self.layers_down = nn.ModuleList([
            vss_layer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                fmap_size=fmas[i_layer],
                stage_spec=stage_specs[i_layer],
            ) for i_layer in range(self.num_layers)
        ])

        fmas = [56, 28, 14, 7][::-1]
        self.layers_up = nn.ModuleList([
            vss_layer(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
                fmap_size=fmas[i_layer],
                stage_spec=stage_specs[::-1][i_layer],
            ) for i_layer in range(self.num_layers)
        ])

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1]//4, num_classes, 1)
        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        skip_list = []
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers_down:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list

    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up):
            x = layer_up(x) if inx == 0 else layer_up(x+skip_list[-inx])
        return x

    def forward_final(self, x):
        x = self.final_up(x)
        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers_down:
            x = layer(x)
        return x

    def forward(self, x):
        x, skip_list = self.forward_features(x)
        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)
        return x

