from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from tqdm import trange

from functools import partial
# Import vision transformer components
from vit.vision_transformer import TriplaneFusionBlockv4_nested, TriplaneFusionBlockv4_nested_init_from_dino_lite, TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout, VisionTransformer, TriplaneFusionBlockv4_nested_init_from_dino

# Import triplane components
from nsr.triplane import OSGDecoder, Triplane, Triplane_fg_bg_plane

from .vision_transformer import Block, VisionTransformer
from .utils import trunc_normal_

from guided_diffusion import dist_util, logger

from ipdb import set_trace as st

# Import modules for diffusion and upsampling
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from utils.torch_utils.components import PixelShuffleUpsample, ResidualBlock, Upsample, PixelUnshuffleUpsample, Conv3x3TriplaneTransformation
from utils.torch_utils.distributions.distributions import DiagonalGaussianDistribution
from nsr.superresolution import SuperresolutionHybrid2X, SuperresolutionHybrid4X

from torch.nn.parameter import Parameter, UninitializedParameter, UninitializedBuffer

from .vision_transformer import *

# Import position embedding and other utilities
from dit.dit_models import get_2d_sincos_pos_embed
from torch import _assert
from itertools import repeat
import collections.abc
from .quant import VectorQuantizer2

import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from nsr.triplane import TriplaneMesh

# Helper function to convert single value to n-tuple
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


class PatchEmbedTriplane(nn.Module):
    """ Group convolution patch embedder for triplane features
    
    Args:
        img_size: Input image size
        patch_size: Size of patches to embed
        in_chans: Number of input channels
        embed_dim: Embedding dimension
        norm_layer: Normalization layer
        flatten: Whether to flatten spatial dimensions
        bias: Whether to use bias in convolution
    """

    def __init__(
        self,
        img_size=32,
        patch_size=2,
        in_chans=4,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        patch_size = (3,3)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # Group convolution - each group processes one plane
        self.proj = nn.Conv2d(in_chans,
                              embed_dim * 3,
                              kernel_size=patch_size,
                              stride=1,
                              padding=1,
                              bias=bias,
                              groups=3)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(
            H == self.img_size[0],
            f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        )
        _assert(
            W == self.img_size[1],
            f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        )

        # Apply group convolution to get embeddings
        x = self.proj(x)  # B 3*C token_H token_W

        # Reshape to separate planes
        x = x.reshape(B, x.shape[1] // 3, 3, x.shape[-2],
                      x.shape[-1])  # B C 3 H W

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BC3HW -> B 3HW C

        x = self.norm(x)
        return x


class PatchEmbedTriplaneRodin(PatchEmbedTriplane):
    """Variant of PatchEmbedTriplane using RodinRollOutConv3D_GroupConv"""

    def __init__(self,
                 img_size=32,
                 patch_size=2,
                 in_chans=4,
                 embed_dim=768,
                 norm_layer=None,
                 flatten=True,
                 bias=True):
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer,
                         flatten, bias)
        self.proj = RodinRollOutConv3D_GroupConv(in_chans,
                                                 embed_dim * 3,
                                                 kernel_size=patch_size,
                                                 stride=patch_size,
                                                 padding=0)


class ViTTriplaneDecomposed(nn.Module):
    """Vision Transformer for triplane feature processing"""
    def __init__(
            self,
            vit_decoder,
            triplane_decoder: Triplane,
            cls_token=False,
            decoder_pred_size=-1,
            unpatchify_out_chans=-1,
            channel_multiplier=4,
            use_fusion_blk=True,
            fusion_blk_depth=4,
            fusion_blk=TriplaneFusionBlock,
            fusion_blk_start=0,
            ldm_z_channels=4,
            ldm_embed_dim=4,
            vae_p=2,
            token_size=None,
            w_avg=torch.zeros([512]),
            patch_size=None,
            **kwargs,
    ) -> None:
        super().__init__()
        
        # Initialize model components
        self.superresolution = nn.ModuleDict({})
        self.decomposed_IN = False
        self.decoder_pred_3d = None
        self.transformer_3D_blk = None
        self.logvar = None
        self.channel_multiplier = channel_multiplier
        self.cls_token = cls_token
        self.vit_decoder = vit_decoder
        self.triplane_decoder = triplane_decoder

        # Set patch size
        if patch_size is None:
            self.patch_size = self.vit_decoder.patch_embed.patch_size
        else:
            self.patch_size = patch_size

        if isinstance(self.patch_size, tuple):
            self.patch_size = self.patch_size[0]

        # Set output channels
        if unpatchify_out_chans == -1:
            self.unpatchify_out_chans = self.triplane_decoder.out_chans
        else:
            self.unpatchify_out_chans = unpatchify_out_chans

        # Initialize decoder prediction layer
        if decoder_pred_size == -1:
            decoder_pred_size = self.patch_size**2 * self.triplane_decoder.out_chans

        self.decoder_pred = nn.Linear(
            self.vit_decoder.embed_dim,
            decoder_pred_size,
            bias=True)

        # Set model parameters
        self.plane_n = 3
        self.ldm_z_channels = ldm_z_channels
        self.ldm_embed_dim = ldm_embed_dim
        self.vae_p = vae_p
        self.token_size = 16
        self.vae_res = self.vae_p * self.token_size

        # Initialize positional embeddings
        self.vit_decoder.pos_embed = nn.Parameter(
            torch.zeros(1, 3 * (self.token_size**2 + self.cls_token),
                       vit_decoder.embed_dim))

        # Setup fusion blocks
        self.fusion_blk_start = fusion_blk_start
        self.create_fusion_blks(fusion_blk_depth, use_fusion_blk, fusion_blk)

        # Register average latent vector
        self.register_buffer('w_avg', w_avg)
        self.rendering_kwargs = self.triplane_decoder.rendering_kwargs

    @torch.inference_mode()
    def forward_points(self, planes, points: torch.Tensor, chunk_size: int = 2**16):
        N, P = points.shape[:2]
        if planes.ndim == 4:
            planes = planes.reshape(len(planes), 3, -1, planes.shape[-2], planes.shape[-1])

        outs = []
        for i in trange(0, points.shape[1], chunk_size):
            chunk_points = points[:, i:i+chunk_size]
            chunk_out = self.triplane_decoder.renderer._run_model(
                planes=planes,
                decoder=self.triplane_decoder.decoder,
                sample_coordinates=chunk_points,
                sample_directions=torch.zeros_like(chunk_points),
                options=self.rendering_kwargs,
            )
            outs.append(chunk_out)
            torch.cuda.empty_cache()

        point_features = {
            k: torch.cat([out[k] for out in outs], dim=1)
            for k in outs[0].keys()
        }
        return point_features

    def triplane_decode_grid(self, vit_decode_out, grid_size, aabb: torch.Tensor = None, **kwargs):
        assert isinstance(vit_decode_out, dict)
        planes = vit_decode_out['latent_after_vit']

        if aabb is None:
            if 'sampler_bbox_min' in self.rendering_kwargs:
                aabb = torch.tensor([
                    [self.rendering_kwargs['sampler_bbox_min']] * 3,
                    [self.rendering_kwargs['sampler_bbox_max']] * 3,
                ], device=planes.device, dtype=planes.dtype).unsqueeze(0).repeat(planes.shape[0], 1, 1)
            else:
                aabb = torch.tensor([
                    [-self.rendering_kwargs['box_warp']/2] * 3,
                    [self.rendering_kwargs['box_warp']/2] * 3,
                ], device=planes.device, dtype=planes.dtype).unsqueeze(0).repeat(planes.shape[0], 1, 1)

        assert planes.shape[0] == aabb.shape[0]
        N = planes.shape[0]

        grid_points = []
        for i in range(N):
            grid_points.append(torch.stack(torch.meshgrid(
                torch.linspace(aabb[i, 0, 0], aabb[i, 1, 0], grid_size, device=planes.device),
                torch.linspace(aabb[i, 0, 1], aabb[i, 1, 1], grid_size, device=planes.device),
                torch.linspace(aabb[i, 0, 2], aabb[i, 1, 2], grid_size, device=planes.device),
                indexing='ij',
            ), dim=-1).reshape(-1, 3))
        cube_grid = torch.stack(grid_points, dim=0).to(planes.device)

        features = self.forward_points(planes, cube_grid)

        features = {
            k: v.reshape(N, grid_size, grid_size, grid_size, -1)
            for k, v in features.items()
        }

        return features

    def create_uvit_arch(self):
        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2:]:
            blk.skip_linear = nn.Linear(2 * self.vit_decoder.embed_dim,
                                      self.vit_decoder.embed_dim)

            nn.init.constant_(blk.skip_linear.weight, 0)
            if isinstance(blk.skip_linear, nn.Linear) and blk.skip_linear.bias is not None:
                nn.init.constant_(blk.skip_linear.bias, 0)


    def vit_decode_backbone(self, latent, img_size):
        return self.forward_vit_decoder(latent, img_size)

    def init_weights(self):
        # Initialize pos_embed with sin-cos embedding
        p = self.token_size
        D = self.vit_decoder.pos_embed.shape[-1]
        grid_size = (3 * p, p)
        pos_embed = get_2d_sincos_pos_embed(D, grid_size).reshape(3 * p * p, D)
        self.vit_decoder.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def create_fusion_blks(self, fusion_blk_depth, use_fusion_blk, fusion_blk):
        vit_decoder_blks = self.vit_decoder.blocks
        assert len(vit_decoder_blks) == 12, 'ViT-B by default'

        nh = self.vit_decoder.blocks[0].attn.num_heads
        dim = self.vit_decoder.embed_dim

        fusion_blk_start = self.fusion_blk_start
        triplane_fusion_vit_blks = nn.ModuleList()

        if fusion_blk_start != 0:
            for i in range(0, fusion_blk_start):
                triplane_fusion_vit_blks.append(
                    vit_decoder_blks[i])  # append all vit blocks in the front

        for i in range(fusion_blk_start, len(vit_decoder_blks), fusion_blk_depth):
            vit_blks_group = vit_decoder_blks[i:i + fusion_blk_depth]
            triplane_fusion_vit_blks.append(fusion_blk(vit_blks_group, nh, dim, use_fusion_blk))

        self.vit_decoder.blocks = triplane_fusion_vit_blks

    def triplane_decode(self, latent, c):
        ret_dict = self.triplane_decoder(latent, c)
        ret_dict['latent'] = latent
        return ret_dict

    def triplane_renderer(self, latent, coordinates, directions):
        planes = latent.view(len(latent), 3,
                           self.triplane_decoder.decoder_in_chans,
                           latent.shape[-2],
                           latent.shape[-1])

        return self.triplane_decoder.renderer.run_model(
            planes, 
            self.triplane_decoder.decoder,
            coordinates, 
            directions,
            self.triplane_decoder.rendering_kwargs
        )

    def unpatchify_triplane(self, x, p=None, unpatchify_out_chans=None):
        if unpatchify_out_chans is None:
            unpatchify_out_chans = self.unpatchify_out_chans // 3
            
        if self.cls_token:
            x = x[:, 1:]

        if p is None:
            p = self.patch_size
            
        h = w = int((x.shape[1] // 3)**.5)
        assert h * w * 3 == x.shape[1]

        x = x.reshape(shape=(x.shape[0], 3, h, w, p, p, unpatchify_out_chans))
        x = torch.einsum('ndhwpqc->ndchpwq', x)
        triplanes = x.reshape(shape=(x.shape[0], unpatchify_out_chans * 3, h * p, h * p))
        return triplanes

    def interpolate_pos_encoding(self, x, w, h):
        return self.vit_decoder.pos_embed

    def forward_vit_decoder(self, x, img_size=None):
        if img_size is None:
            img_size = self.img_size

        if self.cls_token:
            x = x + self.vit_decoder.interpolate_pos_encoding(x, img_size, img_size)[:, :]
        else:
            x = x + self.vit_decoder.interpolate_pos_encoding(x, img_size, img_size)[:, 1:]

        for blk in self.vit_decoder.blocks:
            x = blk(x)
        x = self.vit_decoder.norm(x)

        return x

    def unpatchify(self, x, p=None, unpatchify_out_chans=None):
        if unpatchify_out_chans is None:
            unpatchify_out_chans = self.unpatchify_out_chans
            
        if self.cls_token:
            x = x[:, 1:]

        if p is None:
            p = self.patch_size
            
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, unpatchify_out_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], unpatchify_out_chans, h * p, h * p))
        return imgs

    def forward(self, latent, c, img_size):
        latent = self.forward_vit_decoder(latent, img_size)

        cls_token = latent[:, :1] if self.cls_token else None

        latent = self.decoder_pred(latent)
        latent = self.unpatchify(latent)

        ret_dict = self.triplane_decoder(planes=latent, c=c)
        ret_dict.update({'latent': latent, 'cls_token': cls_token})

        return ret_dict

class RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn(
        ViTTriplaneDecomposed):
    # Lite version without sd-bg, using TriplaneFusionBlockv4_nested_init_from_dino
    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino,
            channel_multiplier=4,
            ldm_z_channels=4,
            ldm_embed_dim=4,
            vae_p=2,
            **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            channel_multiplier=channel_multiplier,
            use_fusion_blk=use_fusion_blk,
            fusion_blk_depth=fusion_blk_depth,
            fusion_blk=fusion_blk,
            ldm_z_channels=ldm_z_channels,
            ldm_embed_dim=ldm_embed_dim,
            vae_p=vae_p,
            decoder_pred_size=(4 // 1)**2 *
            int(triplane_decoder.out_chans // 3 * channel_multiplier),
            **kwargs)

        # Latent VAE modules
        Cvae = 8  # Codebook size
        self.superresolution.update(
            dict(
                ldm_downsample=nn.Linear(
                    384,
                    self.vae_p * self.vae_p * 3 * self.ldm_z_channels *
                    2,  # 48
                    bias=True),
                ldm_upsample=PatchEmbedTriplane(
                    16,
                    self.vae_p,
                    3 * Cvae,
                    vit_decoder.embed_dim,
                    bias=True),
                quant_conv=nn.Conv2d(2 * 3 * self.ldm_z_channels,
                                     2 * self.ldm_embed_dim * 3,
                                     kernel_size=1,
                                     groups=3),
                conv_sr=RodinConv3D4X_lite_mlp_as_residual_lite(
                    int(triplane_decoder.out_chans * channel_multiplier),
                    int(triplane_decoder.out_chans * 1))))

        # Initialize weights
        self.init_weights()
        self.reparameterization_soft_clamp = True  # Some instability in training VAE

        self.create_uvit_arch()

        # Multi-Scale VQ initialization
        vocab_size = 16384  # Codebook size
        using_znorm = True
        beta = 0.25
        default_qresi_counts = 0
        v_patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)  # Scale list
        quant_resi = 0.5
        share_quant_resi = 4
        quant_conv_ks = 3

        self.superresolution.update(
            dict(quant_conv=torch.nn.Conv2d(Cvae, Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2),
            post_quant_conv = torch.nn.Conv2d(Cvae, Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2),
         quantize= VectorQuantizer2(
            vocab_size=vocab_size, Cvae=Cvae, using_znorm=using_znorm, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi, share_quant_resi=share_quant_resi,
        )   
            ))

    def vit_decode(self, latent, img_size, sample_posterior=True):
        ret_dict = self.vae_reparameterization(latent, sample_posterior)
        latent, usages, vq_loss = self.vit_decode_backbone(ret_dict, img_size)
        return self.vit_decode_postprocess(latent, ret_dict), vq_loss, usages

    def unpatchify3D(self, x, p, unpatchify_out_chans, plane_n=3):
        """
        x: (N, L, patch_size**2 * self.out_chans)
        returns: 3D latents
        """

        if self.cls_token:
            x = x[:, 1:]

        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, plane_n,
                             unpatchify_out_chans))

        x = torch.einsum('nhwpqdc->ndhpwqc', x)

        latents3D = x.reshape(shape=(x.shape[0], plane_n, h * p, h * p,
                                     unpatchify_out_chans))
        return latents3D

    def vae_encode(self, h):
        B, _, H, W = h.shape
        moments = self.superresolution['quant_conv'](h)

        # Reshape to 3 latent planes
        moments = moments.reshape(
            B,
            moments.shape[1] // self.plane_n,
            self.plane_n,
            H,
            W,
        )  # B C 3 H W

        moments = moments.flatten(-2)  # B C 3 L

        # Extract mean and logvar
        posterior = DiagonalGaussianDistribution(
            moments, soft_clamp=self.reparameterization_soft_clamp)
        return posterior

    def vae_reparameterization(self, latent, sample_posterior):
        """
        Input: latent from ViT encoder
        """
        # First downsample for VAE
        latents3D = self.superresolution['ldm_downsample'](latent)

        assert self.vae_p > 1
        latents3D = self.unpatchify3D(
            latents3D,
            p=self.vae_p,
            unpatchify_out_chans=self.ldm_z_channels * 2)

        B, _, H, W, C = latents3D.shape
        latents3D = latents3D.permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)

        # Perform VAE encoding
        posterior = self.vae_encode(latents3D)

        if sample_posterior:
            latent = posterior.sample()
        else:
            latent = posterior.mode()

        log_q = posterior.log_p(latent)

        # For LSGM KL code
        latent_normalized_2Ddiffusion = latent.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)
        log_q_2Ddiffusion = log_q.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)

        latent = latent.permute(0, 2, 3, 1)
        latent = latent.reshape(latent.shape[0], -1, latent.shape[-1])

        ret_dict = dict(
            normal_entropy=posterior.normal_entropy(),
            latent_normalized=latent,
            latent_normalized_2Ddiffusion=latent_normalized_2Ddiffusion,
            log_q_2Ddiffusion=log_q_2Ddiffusion,
            log_q=log_q,
            posterior=posterior,
        )

        return ret_dict

    def vit_decode_backbone(self, latent, img_size):
        if isinstance(latent, dict):
            latent = latent['latent_normalized_2Ddiffusion']

        latent = self.superresolution['ldm_upsample'](latent)

        return self.forward_vit_decoder(latent, img_size)

    def triplane_decode(self,
                        vit_decode_out,
                        c,
                        return_raw_only=False,
                        **kwargs):
        if isinstance(vit_decode_out, dict):
            latent_after_vit, sr_w_code = (vit_decode_out.get(k, None)
                                           for k in ('latent_after_vit',
                                                     'sr_w_code'))
        else:
            latent_after_vit = vit_decode_out
            sr_w_code = None
            vit_decode_out = dict(latent_normalized=latent_after_vit)

        ret_dict = self.triplane_decoder(latent_after_vit,
                                         c,
                                         ws=sr_w_code,
                                         return_raw_only=return_raw_only,
                                         **kwargs)

        ret_dict.update({
            'latent_after_vit': latent_after_vit,
            **vit_decode_out
        })

        return ret_dict

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):
        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection
        latent = self.decoder_pred(latent_from_vit)

        latent = self.unpatchify_triplane(
            latent,
            p=4,
            unpatchify_out_chans=int(
                self.channel_multiplier * self.unpatchify_out_chans // 3))

        # 4X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr'](latent)

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent_from_vit.shape[0], 0)))

        return ret_dict

    def forward_vit_decoder(self, x, img_size=None):
        if img_size is None:
            img_size = self.img_size

        x = x + self.interpolate_pos_encoding(x, img_size, img_size)

        B, L, C = x.shape
        x = x.view(B, 3, L // 3, C)

        skips = [x]
        assert self.fusion_blk_start == 0

        # Encoder blocks
        for blk in self.vit_decoder.blocks[0:len(self.vit_decoder.blocks) // 2 - 1]:
            x = blk(x)
            skips.append(x)

        # Middle blocks
        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2 - 1:len(self.vit_decoder.blocks) // 2]:
            x = blk(x)

        # Decoder blocks with skip connections
        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2:]:
            x = x + blk.skip_linear(torch.cat([x, skips.pop()], dim=-1))
            x = blk(x)

        x = self.vit_decoder.norm(x)
        x = x.view(B, L, C)
        return x


# SD version
class RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn_SD(
        RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane_fg_bg_plane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         use_fusion_blk=use_fusion_blk,
                         fusion_blk_depth=fusion_blk_depth,
                         fusion_blk=fusion_blk,
                         channel_multiplier=channel_multiplier,
                         **kwargs)

        # Remove unused components
        for k in [
                'ldm_downsample',
        ]:
            del self.superresolution[k]

    def vae_reparameterization(self, latent, sample_posterior):
        # latent: B 24 32 32

        assert self.vae_p > 1

        # Do VAE here
        posterior = self.vae_encode(latent)  # B self.ldm_z_channels 3 L

        if sample_posterior:
            latent = posterior.sample()
        else:
            latent = posterior.mode()  # B C 3 L

        log_q = posterior.log_p(latent)  # same shape as latent

        # For LSGM KL code
        latent_normalized_2Ddiffusion = latent.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, 3*4, 16 16
        log_q_2Ddiffusion = log_q.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, 3*4, 16 16

        latent = latent.permute(0, 2, 3, 1)  # B C 3 L -> B 3 L C

        latent = latent.reshape(latent.shape[0], -1,
                                latent.shape[-1])  # B 3*L C

        ret_dict = dict(
            normal_entropy=posterior.normal_entropy(),
            latent_normalized=latent,
            latent_normalized_2Ddiffusion=latent_normalized_2Ddiffusion,  # 
            log_q_2Ddiffusion=log_q_2Ddiffusion,
            log_q=log_q,
            posterior=posterior,
        )

        return ret_dict

class RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn_SD_D(
        RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn_SD):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane_fg_bg_plane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        self.decoder_pred = None

        self.superresolution.update(
            dict(conv_sr=Decoder(
                resolution=128,
                in_channels=3,
                ch=32,
                ch_mult=[1, 2, 2, 4],
                num_res_blocks=1,
                dropout=0.0,
                attn_resolutions=[],
                out_ch=32,
                z_channels=vit_decoder.embed_dim,
            )))

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):
        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        def unflatten_token(x, p=None):
            B, L, C = x.shape
            x = x.reshape(B, 3, L // 3, C)

            if self.cls_token:
                x = x[:, :, 1:]

            h = w = int((x.shape[2])**.5)
            assert h * w == x.shape[2]

            if p is None:
                x = x.reshape(shape=(B, 3, h, w, -1))
                x = rearrange(x, 'b n h w c->(b n) c h w')
            else:
                x = x.reshape(shape=(B, 3, h, w, p, p, -1))
                x = rearrange(x, 'b n h w p1 p2 c->(b n) c (h p1) (w p2)')

            return x

        latent = unflatten_token(latent_from_vit)
        latent = self.superresolution['conv_sr'](latent)
        latent = rearrange(latent, '(b n) c h w->b (n c) h w', n=3)

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        return ret_dict


class RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_lite3DAttn(
        RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn_SD):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane_fg_bg_plane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        self.decoder_pred = nn.Linear(self.vit_decoder.embed_dim // 3,
                                      2048,
                                      bias=True)

        self.superresolution.update(
            dict(ldm_upsample=PatchEmbedTriplaneRodin(
                self.vae_p * self.token_size,
                self.vae_p,
                3 * self.ldm_embed_dim,
                vit_decoder.embed_dim // 3,
                bias=True)))

        has_token = bool(self.cls_token)
        self.vit_decoder.pos_embed = nn.Parameter(
            torch.zeros(1, 16 * 16 + has_token, vit_decoder.embed_dim))

    def forward(self, latent, c, img_size):
        latent_normalized = self.vit_decode(latent, img_size)
        return self.triplane_decode(latent_normalized, c)

    def vae_reparameterization(self, latent, sample_posterior):
        assert self.vae_p > 1

        posterior = self.vae_encode(latent)

        if sample_posterior:
            latent = posterior.sample()
        else:
            latent = posterior.mode()

        log_q = posterior.log_p(latent)

        latent_normalized_2Ddiffusion = latent.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)
        log_q_2Ddiffusion = log_q.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)

        latent = latent.permute(0, 3, 1, 2)
        latent = latent.reshape(*latent.shape[:2], -1)

        ret_dict = dict(
            normal_entropy=posterior.normal_entropy(),
            latent_normalized=latent,
            latent_normalized_2Ddiffusion=latent_normalized_2Ddiffusion,
            log_q_2Ddiffusion=log_q_2Ddiffusion,
            log_q=log_q,
            posterior=posterior,
        )

        return ret_dict

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):
        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        B, N, C = latent_from_vit.shape
        latent_from_vit = latent_from_vit.reshape(B, N, C // 3, 3).permute(
            0, 3, 1, 2)

        latent = self.decoder_pred(latent_from_vit)
        latent = latent.reshape(B, 3 * N, -1)

        latent = self.unpatchify_triplane(
            latent,
            p=4,
            unpatchify_out_chans=int(
                self.channel_multiplier * self.unpatchify_out_chans // 3))

        latent = self.superresolution['conv_sr'](latent)

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent_from_vit.shape[0], 0)))

        return ret_dict

    def vit_decode_backbone(self, latent, img_size):
        if isinstance(latent, dict):
            latent = latent['latent_normalized_2Ddiffusion']

        latent = self.superresolution['ldm_upsample'](latent)

        B, N3, C = latent.shape
        latent = latent.reshape(B, 3, N3 // 3, C).permute(0, 2, 3, 1)
        latent = latent.reshape(*latent.shape[:2], -1)

        return self.forward_vit_decoder(latent, img_size)

    def forward_vit_decoder(self, x, img_size=None):
        if img_size is None:
            img_size = self.img_size

        # if self.cls_token:
        x = x + self.interpolate_pos_encoding(x, img_size,
                                              img_size)[:, :]  # B, L, C

        B, L, C = x.shape

        skips = [x]
        assert self.fusion_blk_start == 0

        for blk in self.vit_decoder.blocks[0:len(self.vit_decoder.blocks) // 2 - 1]:
            x = blk(x)
            skips.append(x)

        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2 - 1:len(self.vit_decoder.blocks) // 2]:
            x = blk(x)

        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2:]:
            x = x + blk.skip_linear(torch.cat([x, skips.pop()], dim=-1))
            x = blk(x)

        x = self.vit_decoder.norm(x)
        x = x.view(B, L, C)
        return x

    def create_fusion_blks(self, fusion_blk_depth, use_fusion_blk, fusion_blk):
        vit_decoder_blks = self.vit_decoder.blocks
        assert len(vit_decoder_blks) == 12

        nh = self.vit_decoder.blocks[0].attn.num_heads // 3
        dim = self.vit_decoder.embed_dim // 3

        fusion_blk_start = self.fusion_blk_start
        triplane_fusion_vit_blks = nn.ModuleList()

        if fusion_blk_start != 0:
            for i in range(0, fusion_blk_start):
                triplane_fusion_vit_blks.append(vit_decoder_blks[i])

        for i in range(fusion_blk_start, len(vit_decoder_blks), fusion_blk_depth):
            vit_blks_group = vit_decoder_blks[i:i + fusion_blk_depth]
            triplane_fusion_vit_blks.append(
                fusion_blk(vit_blks_group, nh, dim, use_fusion_blk))

        self.vit_decoder.blocks = triplane_fusion_vit_blks

# default for objaverse
class RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_S(
        RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            use_fusion_blk=use_fusion_blk,
            fusion_blk_depth=fusion_blk_depth,
            fusion_blk=fusion_blk,
            channel_multiplier=channel_multiplier,
            **kwargs)
        self.D_roll_out_input = False

        for k in ['ldm_downsample']:
            del self.superresolution[k]

        self.decoder_pred = None
        self.superresolution.update(
            dict(
                conv_sr=Decoder(
                    resolution=128,
                    in_channels=3,
                    ch=32,
                    ch_mult=[1, 1, 2, 2, 4],
                    num_res_blocks=2,
                    dropout=0.0,
                    attn_resolutions=[],
                    out_ch=32,
                    z_channels=vit_decoder.embed_dim,
                ),
            ))

        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2:]:
            del blk.skip_linear
        
        self.dtype = torch.bfloat16

    @torch.inference_mode()
    def forward_points(self,
                       planes,
                       points: torch.Tensor,
                       chunk_size: int = 2**16):
        N, P = points.shape[:2]
        if planes.ndim == 4:
            planes = planes.reshape(
                len(planes),
                3,
                -1,
                planes.shape[-2],
                planes.shape[-1])

        outs = []
        for i in trange(0, points.shape[1], chunk_size):
            chunk_points = points[:, i:i + chunk_size]

            chunk_out = self.triplane_decoder.renderer._run_model(
                planes=planes,
                decoder=self.triplane_decoder.decoder,
                sample_coordinates=chunk_points,
                sample_directions=torch.zeros_like(chunk_points),
                options=self.rendering_kwargs,
            )

            outs.append(chunk_out)
            torch.cuda.empty_cache()

        point_features = {
            k: torch.cat([out[k] for out in outs], dim=1)
            for k in outs[0].keys()
        }
        return point_features

    def triplane_decode_grid(self,
                             vit_decode_out,
                             grid_size,
                             aabb: torch.Tensor = None,
                             **kwargs):
        assert isinstance(vit_decode_out, dict)
        planes = vit_decode_out['latent_after_vit']

        if aabb is None:
            if 'sampler_bbox_min' in self.rendering_kwargs:
                aabb = torch.tensor([
                    [self.rendering_kwargs['sampler_bbox_min']] * 3,
                    [self.rendering_kwargs['sampler_bbox_max']] * 3,
                ],
                                    device=planes.device,
                                    dtype=planes.dtype).unsqueeze(0).repeat(
                                        planes.shape[0], 1, 1)
            else:
                aabb = torch.tensor([
                    [-self.rendering_kwargs['box_warp'] / 2] * 3,
                    [self.rendering_kwargs['box_warp'] / 2] * 3,
                ],
                    device=planes.device,
                    dtype=planes.dtype).unsqueeze(0).repeat(
                        planes.shape[0], 1, 1)

        assert planes.shape[0] == aabb.shape[0]
        N = planes.shape[0]

        grid_points = []
        for i in range(N):
            grid_points.append(
                torch.stack(torch.meshgrid(
                    torch.linspace(aabb[i, 0, 0],
                                   aabb[i, 1, 0],
                                   grid_size,
                                   device=planes.device),
                    torch.linspace(aabb[i, 0, 1],
                                   aabb[i, 1, 1],
                                   grid_size,
                                   device=planes.device),
                    torch.linspace(aabb[i, 0, 2],
                                   aabb[i, 1, 2],
                                   grid_size,
                                   device=planes.device),
                    indexing='ij',
                ),
                            dim=-1).reshape(-1, 3))
        cube_grid = torch.stack(grid_points, dim=0).to(planes.device)

        features = self.forward_points(planes, cube_grid)

        features = {
            k: v.reshape(N, grid_size, grid_size, grid_size, -1)
            for k, v in features.items()
        }

        return features

    def create_fusion_blks(self, fusion_blk_depth, use_fusion_blk, fusion_blk):
        pass

    def forward_vit_decoder(self, x, img_size=None):
        return self.vit_decoder(x)

    def vit_decode_backbone(self, latent, img_size):
        if isinstance(latent, dict):
            latent = latent['latent_normalized_2Ddiffusion']

        latent = self.superresolution['quant_conv'](latent)

        f_hat, usages, vq_loss = self.superresolution['quantize'](latent, ret_usages=True)

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            f_hat = self.superresolution['post_quant_conv'](f_hat.to(self.dtype))
            f_hat = f_hat.reshape(f_hat.shape[0] // 3, -1, f_hat.shape[-2], f_hat.shape[-1])
            f_hat = self.superresolution['ldm_upsample'](f_hat)
            f_hat = self.forward_vit_decoder(f_hat, img_size)

            return f_hat, usages, vq_loss

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):
        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        def unflatten_token(x, p=None):
            B, L, C = x.shape
            x = x.reshape(B, 3, L // 3, C)

            if self.cls_token:
                x = x[:, :, 1:]
            h = w = int((x.shape[2])**.5)
            assert h * w == x.shape[2]

            if p is None:
                x = x.reshape(shape=(B, 3, h, w, -1))
                if not self.D_roll_out_input:
                    x = rearrange(x, 'b n h w c->(b n) c h w')
                else:
                    x = rearrange(x, 'b n h w c->b c h (n w)')
            else:
                x = x.reshape(shape=(B, 3, h, w, p, p, -1))
                if self.D_roll_out_input:
                    x = rearrange(x, 'b n h w p1 p2 c->b c (h p1) (n w p2)')
                else:
                    x = rearrange(x, 'b n h w p1 p2 c->(b n) c (h p1) (w p2)')

            return x

        latent = unflatten_token(latent_from_vit)
        latent = self.superresolution['conv_sr'](latent)

        if not self.D_roll_out_input:
            latent = rearrange(latent, '(b n) c h w->b (n c) h w', n=3)
        else:
            latent = rearrange(latent, 'b c h (n w)->b (n c) h w', n=3)

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        return ret_dict

    def vae_reparameterization(self, latent, sample_posterior):
        assert self.vae_p > 1

        latent_normalized_2Ddiffusion = latent.reshape(
            latent.shape[0] * 3, -1, latent.shape[-2],
            latent.shape[-1])
        log_q_2Ddiffusion = None

        latent = latent.permute(0, 2, 3, 1)
        latent = latent.reshape(latent.shape[0], -1, latent.shape[-1])

        ret_dict = dict(
            normal_entropy=None,
            latent_normalized=latent,
            latent_normalized_2Ddiffusion=latent_normalized_2Ddiffusion,
            log_q_2Ddiffusion=log_q_2Ddiffusion,
            log_q=None,
            posterior=None,
        )

        return ret_dict

    def vit_decode(self, latent, img_size, sample_posterior=True, **kwargs):
        return super().vit_decode(latent, img_size, sample_posterior)

# Default model for Objaverse dataset
class ft(RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_S):
    """Fine-tuning model with DiT decoder and SR decoder"""
    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            use_fusion_blk=use_fusion_blk,
            fusion_blk_depth=fusion_blk_depth,
            fusion_blk=fusion_blk,
            channel_multiplier=channel_multiplier,
            patch_size=-1,  # Placeholder since using DiT
            token_size=2,
            **kwargs)

        self.superresolution.update(
            dict(
                conv_sr=Decoder(
                    resolution=128,
                    in_channels=3,
                    ch=32,
                    ch_mult=[1,2,2,4],
                    num_res_blocks=1,
                    dropout=0.0,
                    attn_resolutions=[],
                    out_ch=32,
                    z_channels=vit_decoder.embed_dim,
                ),
            ))


# Base model for Objaverse
class RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout(
        RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn_SD):
    """Base model with 3D attention and rollout"""
    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)


# Final version with SD decoder
class RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D(
        RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout):
    """Final model with Stable Diffusion decoder"""
    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        self.decoder_pred = None  # Directly un-patchembed
        self.superresolution.update(
            dict(
                conv_sr=Decoder(
                    resolution=128,
                    in_channels=3,
                    ch=32,
                    ch_mult=[1, 2, 2, 4],
                    num_res_blocks=1,
                    dropout=0.0,
                    attn_resolutions=[],
                    out_ch=32,
                    z_channels=vit_decoder.embed_dim,
                ),
            ))
        self.D_roll_out_input = False

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):
        """Post-process ViT decoder output"""
        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        def unflatten_token(x, p=None):
            """Unflatten tokens into spatial dimensions"""
            B, L, C = x.shape
            x = x.reshape(B, 3, L // 3, C)

            if self.cls_token:
                x = x[:, :, 1:]  # Remove CLS token

            h = w = int((x.shape[2])**.5)
            assert h * w == x.shape[2]

            if p is None:
                x = x.reshape(shape=(B, 3, h, w, -1))
                if not self.D_roll_out_input:
                    x = rearrange(x, 'b n h w c->(b n) c h w')
                else:
                    x = rearrange(x, 'b n h w c->b c h (n w)')
            else:
                x = x.reshape(shape=(B, 3, h, w, p, p, -1))
                if self.D_roll_out_input:
                    x = rearrange(x, 'b n h w p1 p2 c->b c (h p1) (n w p2)')
                else:
                    x = rearrange(x, 'b n h w p1 p2 c->(b n) c (h p1) (w p2)')

            return x

        latent = unflatten_token(latent_from_vit)
        latent = self.superresolution['conv_sr'](latent)

        if not self.D_roll_out_input:
            latent = rearrange(latent, '(b n) c h w->b (n c) h w', n=3)
        else:
            latent = rearrange(latent, 'b c h (n w)->b (n c) h w', n=3)

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))
        return ret_dict

class RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder(
        RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D
):
    """DIT decoder variant that removes skip connections and adds triplane grid decoding.
    
    This class extends the base model to:
    1. Remove skip connections from decoder blocks
    2. Add methods for decoding triplane features into 3D grid points
    3. Support chunked point sampling to handle memory constraints
    """

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         patch_size=-1,
                         **kwargs)

        # Remove skip connections from decoder blocks
        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2:]:
            del blk.skip_linear

    @torch.inference_mode()
    def forward_points(self,
                       planes,
                       points: torch.Tensor,
                       chunk_size: int = 2**16):
        """Forward pass to decode triplane features at 3D point locations.
        
        Args:
            planes: Triplane features of shape (N, 3, D', H', W')
            points: Query points of shape (N, P, 3) 
            chunk_size: Number of points to process in each chunk
            
        Returns:
            Dictionary of decoded features for each point
        """
        N, P = points.shape[:2]
        if planes.ndim == 4:
            planes = planes.reshape(
                len(planes),
                3,
                -1,
                planes.shape[-2],
                planes.shape[-1])

        # Query triplane in chunks to handle memory
        outs = []
        for i in trange(0, points.shape[1], chunk_size):
            chunk_points = points[:, i:i + chunk_size]
            chunk_out = self.triplane_decoder.renderer._run_model(
                planes=planes,
                decoder=self.triplane_decoder.decoder,
                sample_coordinates=chunk_points,
                sample_directions=torch.zeros_like(chunk_points),
                options=self.rendering_kwargs,
            )
            outs.append(chunk_out)
            torch.cuda.empty_cache()

        point_features = {
            k: torch.cat([out[k] for out in outs], dim=1)
            for k in outs[0].keys()
        }
        return point_features

    def triplane_decode_grid(self,
                             vit_decode_out,
                             grid_size,
                             aabb: torch.Tensor = None,
                             **kwargs):
        """Decode triplane features into a regular 3D grid.
        
        Args:
            vit_decode_out: Dictionary containing latent features
            grid_size: Size of grid to sample
            aabb: Axis-aligned bounding box for sampling. Shape (N, 2, 3)
            
        Returns:
            Dictionary of decoded features on the 3D grid
        """
        assert isinstance(vit_decode_out, dict)
        planes = vit_decode_out['latent_after_vit']

        # Set default AABB if not provided
        if aabb is None:
            if 'sampler_bbox_min' in self.rendering_kwargs:
                aabb = torch.tensor([
                    [self.rendering_kwargs['sampler_bbox_min']] * 3,
                    [self.rendering_kwargs['sampler_bbox_max']] * 3,
                ],
                                    device=planes.device,
                                    dtype=planes.dtype).unsqueeze(0).repeat(
                                        planes.shape[0], 1, 1)
            else:
                aabb = torch.tensor([
                    [-self.rendering_kwargs['box_warp'] / 2] * 3,
                    [self.rendering_kwargs['box_warp'] / 2] * 3,
                ],
                    device=planes.device,
                    dtype=planes.dtype).unsqueeze(0).repeat(
                        planes.shape[0], 1, 1)

        assert planes.shape[0] == aabb.shape[0]
        N = planes.shape[0]

        # Create grid points for sampling
        grid_points = []
        for i in range(N):
            grid_points.append(
                torch.stack(torch.meshgrid(
                    torch.linspace(aabb[i, 0, 0],
                                   aabb[i, 1, 0],
                                   grid_size,
                                   device=planes.device),
                    torch.linspace(aabb[i, 0, 1],
                                   aabb[i, 1, 1],
                                   grid_size,
                                   device=planes.device),
                    torch.linspace(aabb[i, 0, 2],
                                   aabb[i, 1, 2],
                                   grid_size,
                                   device=planes.device),
                    indexing='ij',
                ),
                            dim=-1).reshape(-1, 3))
        cube_grid = torch.stack(grid_points, dim=0).to(planes.device)

        # Decode features at grid points
        features = self.forward_points(planes, cube_grid)
        features = {
            k: v.reshape(N, grid_size, grid_size, grid_size, -1)
            for k, v in features.items()
        }

        return features

    def create_fusion_blks(self, fusion_blk_depth, use_fusion_blk, fusion_blk):
        """No fusion blocks needed in this variant."""
        pass

    def forward_vit_decoder(self, x, img_size=None):
        """Simple forward pass through ViT decoder."""
        return self.vit_decoder(x)

    def vit_decode_backbone(self, latent, img_size):
        return super().vit_decode_backbone(latent, img_size)

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):
        return super().vit_decode_postprocess(latent_from_vit, ret_dict)

    def vae_reparameterization(self, latent, sample_posterior):
        return super().vae_reparameterization(latent, sample_posterior)

