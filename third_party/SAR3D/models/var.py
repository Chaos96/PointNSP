import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import utils.dist as dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn, AdaLNCrossSelfAttn_Image_new, AdaLNCrossSelfAttn_text
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2

from ipdb import set_trace as st
from einops import rearrange
from torch.nn import functional as F
import numpy as np


class SharedAdaLin(nn.Linear):
    """Shared adaptive layer normalization"""
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    """
    Scale Autoregressive model for 3D generation.
    
    Args:
        vae_local: Local VQVAE model
        num_classes: Number of output classes
        depth: Transformer depth
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Drop path rate
        norm_eps: Layer norm epsilon
        shared_aln: Whether to share adaptive layer norm
        cond_drop_rate: Condition dropout rate
        attn_l2_norm: Whether to use L2 norm for attention
        patch_nums: Progressive patch numbers
        flash_if_available: Whether to use flash attention
        fused_if_available: Whether to use fused operations
    """
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., 
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        
        # Model hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.decoder.superresolution.quantize.Cvae, vae_local.decoder.superresolution.quantize.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # Progressive training index
        
        # Progressive patch configuration
        self.patch_nums = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # Input embedding layers
        quant: VectorQuantizer2 = vae_local.decoder.superresolution.quantize
        self.vae_proxy = (vae_local,)
        self.vae_quant_proxy = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # Initialize embeddings
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.pooler_emb = nn.Linear(1024, self.C)
        nn.init.trunc_normal_(self.pooler_emb.weight.data, mean=0, std=init_std)
        nn.init.trunc_normal_(self.pooler_emb.bias.data, mean=0, std=init_std)
        
        # Position embeddings
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # Initialize position, level and plane embeddings
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn*3, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L*3, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        self.plane_embed = nn.Embedding(3, self.C)
        nn.init.trunc_normal_(self.plane_embed.weight.data, mean=0, std=init_std)
        
        # Backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        # Build transformer blocks
        self.blocks = nn.ModuleList([
            AdaLNCrossSelfAttn_Image_new(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        # Print model configuration
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # Initialize attention masks
        d: torch.Tensor = torch.cat([torch.full((pn*pn*3,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L * 3, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L * 3, self.L * 3)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # Initialize plane embeddings
        p = []
        for pn in self.patch_nums:
            p.append(torch.tensor([0]).repeat(pn*pn))
            p.append(torch.tensor([1]).repeat(pn*pn))
            p.append(torch.tensor([2]).repeat(pn*pn))
        p_exp = torch.cat(p).unsqueeze(0)
        plane_1L = p_exp.contiguous()
        self.register_buffer('plane_1L', plane_1L)

        # Output head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    

 
    @torch.no_grad()
    def autoregressive_infer_cfg_3D_VAR_image_l2norm(
        self, B: int,
        dino_image_embeddings: Optional[Union[int, torch.LongTensor]], 
        pooler_output: Optional[int] = None,
        cfg=1.5, top_k=0, top_p=0.0,
        g_seed: Optional[int] = None,
        more_smooth=False,
    ) -> torch.Tensor:
        """
        Autoregressive inference with classifier-free guidance for generating 3D VAR triplane.
        
        Args:
            B: Batch size
            dino_image_embeddings: DINO image embeddings
            pooler_output: Pooled output features
            cfg: Classifier-free guidance ratio
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter 
            g_seed: Random seed
            more_smooth: Whether to use Gumbel softmax for smoother outputs
            
        Returns:
            Tuple of:
            - Generated image latents after VIT decoder
            - Generated token indices
        """
        # Setup random number generator if seed provided
        rng = None if g_seed is None else self.rng.manual_seed(g_seed); rng = self.rng

        # Load empty embeddings for classifier-free guidance
        empty_pooler = torch.from_numpy(np.load("./files/empty_dino_pooler_output.npy")).to(pooler_output.device).unsqueeze(0)
        empty_dino = torch.from_numpy(np.load("./files/empty_dino_embedding.npy"))[1:, :].to(pooler_output.device).unsqueeze(0)
        
        # Concatenate real and empty embeddings for CFG
        pooler_output = torch.cat((pooler_output, empty_pooler.expand(pooler_output.shape)), dim=0)
        dino_image_embeddings = torch.cat((dino_image_embeddings, empty_dino.expand(dino_image_embeddings.shape)), dim=0)

        # Get initial tokens and positional embeddings
        sos = cond_BD = self.pooler_emb(pooler_output)
        sos = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1)
        sos = sos.repeat(1, 3, 1)
        
        # Add level and plane embeddings
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC + self.plane_embed(self.plane_1L)
        next_token_map = sos.expand(2 * B, 3, -1) + lvl_pos[:, :3]
        
        # Initialize feature maps for each plane
        cur_L = 0
        f_hat_list = [
            sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]) for _ in range(3)
        ]

        # Enable KV caching for attention
        for b in self.blocks:
            b.attn.kv_caching(True)

        # Main autoregressive generation loop
        for si, pn in enumerate(self.patch_nums):
            ratio = si / self.num_stages_minus_1
            cur_L += pn * pn * 3

            # Forward pass through transformer blocks
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, dino_condition=dino_image_embeddings, attn_bias=None)
            
            # Get logits and apply classifier-free guidance
            logits_BlV = self.get_logits(x, cond_BD)
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            # Sample tokens
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            g_BL = torch.cat((g_BL, idx_Bl), dim=1) if si > 0 else idx_Bl

            # Get embeddings for sampled tokens
            if not more_smooth:
                embedding = F.normalize(self.vae_quant_proxy[0].embedding.weight, p=2, dim=-1)
                h_BChw_concate = embedding[idx_Bl]
            else:
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                embedding = F.normalize(self.vae_quant_proxy[0].embedding.weight, p=2, dim=-1) 
                h_BChw_concate = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ embedding.unsqueeze(0)

            # Process each plane separately
            h_BChw_list = h_BChw_concate.split(int(h_BChw_concate.shape[1]/3), dim=1)
            next_token_map_list = []
            
            for i in range(3):
                h_BChw = h_BChw_list[i].transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                f_hat_list[i], next_token_map_single = self.vae_quant_proxy[0].get_next_autoregressive_input(
                    si, len(self.patch_nums), f_hat_list[i], h_BChw
                )
                next_token_map_list.append(next_token_map_single.view(B, self.Cvae, -1).transpose(1, 2))

            next_token_map = torch.cat(next_token_map_list, dim=1)

            # Prepare for next stage if not last
            if si != self.num_stages_minus_1:
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + 3 * self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)

        # Disable KV caching
        for b in self.blocks:
            b.attn.kv_caching(False)


        f_hat_all = torch.zeros((3 * f_hat_list[0].shape[0], f_hat_list[0].shape[1], f_hat_list[0].shape[2], f_hat_list[0].shape[3]), device=f_hat_list[0].device, dtype=f_hat_list[0].dtype)
        f_hat_all[0::3] = f_hat_list[0]
        f_hat_all[1::3] = f_hat_list[1] 
        f_hat_all[2::3] = f_hat_list[2]

        # Decode features to image
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            f_hat_all = self.vae_proxy[0].decoder.superresolution['post_quant_conv'](f_hat_all.to(torch.bfloat16))
            f_hat_all = f_hat_all.reshape(f_hat_all.shape[0] // 3, -1, f_hat_all.shape[-2], f_hat_all.shape[-1])
            f_hat_all = self.vae_proxy[0].decoder.superresolution['ldm_upsample'](f_hat_all)
            f_hat_all = self.vae_proxy[0].decoder.forward_vit_decoder(f_hat_all, 224)
            latent_after_vit = self.vit_decode_postprocess(f_hat_all)
            
            return latent_after_vit, g_BL


    @torch.no_grad()
    def reconstruct_gt_Bl_idx(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False, gt_idx: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        Reconstruct image from ground truth token indices
        Args:
            B: batch size
            gt_idx: ground truth token indices
            g_seed: random seed for reproducibility
            more_smooth: use gumbel softmax for smoother results
        Returns:
            Reconstructed image tensor
        """
        # Setup random seed if provided
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng

        # Initialize feature maps for each plane
        f_hat_list = [
            torch.zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1], device=gt_idx.device) 
            for _ in range(3)
        ]
        
        # Enable KV caching for transformer blocks
        for b in self.blocks: b.attn.kv_caching(True)

        # Process each stage
        cur_L = 0
        for si, pn in enumerate(self.patch_nums):
            ratio = si / self.num_stages_minus_1
            cur_L += pn * pn * 3

            # Get token indices for current stage
            idx_Bl = gt_idx[:, cur_L-pn*pn*3:cur_L]

            # Get embeddings for tokens
            if not more_smooth:
                embedding = F.normalize(self.vae_quant_proxy[0].embedding.weight, p=2, dim=-1)
                h_BChw_concate = embedding[idx_Bl]
            else:
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                # Gumbel softmax path not used for evaluation

            # Process each plane separately
            h_BChw_list = h_BChw_concate.split(int(h_BChw_concate.shape[1]/3), dim=1)
            for i in range(3):
                h_BChw = h_BChw_list[i].transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                f_hat_list[i], _ = self.vae_quant_proxy[0].get_next_autoregressive_input(
                    si, len(self.patch_nums), f_hat_list[i], h_BChw
                )

        # Disable KV caching
        for b in self.blocks: b.attn.kv_caching(False)

        f_hat_all = torch.zeros((3 * f_hat_list[0].shape[0], f_hat_list[0].shape[1], f_hat_list[0].shape[2], f_hat_list[0].shape[3]), device=f_hat_list[0].device, dtype=f_hat_list[0].dtype)
        f_hat_all[0::3] = f_hat_list[0]
        f_hat_all[1::3] = f_hat_list[1]
        f_hat_all[2::3] = f_hat_list[2]

        # Decode features to image
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            # Post-process through decoder network
            f_hat_all = self.vae_proxy[0].decoder.superresolution['post_quant_conv'](f_hat_all.to(torch.bfloat16))
            f_hat_all = f_hat_all.reshape(f_hat_all.shape[0] // 3, -1, f_hat_all.shape[-2], f_hat_all.shape[-1]) # (B * 3, 32, 16, 16) -> (B, 32 * 3, 16, 16)

            
            f_hat_all = self.vae_proxy[0].decoder.superresolution['ldm_upsample'](f_hat_all)
            f_hat_all = self.vae_proxy[0].decoder.forward_vit_decoder(f_hat_all, 224)
            latent_after_vit = self.vit_decode_postprocess(f_hat_all)

            return latent_after_vit

    def vit_decode_postprocess(self, latent_from_vit):
        """Post-process ViT decoder output"""
        # Handle classifier token if present
        if self.vae_proxy[0].decoder.cls_token:
            cls_token = latent_from_vit[:, :1]

        # Reshape latent features
        latent = self.unflatten_token(latent_from_vit)
        latent = self.vae_proxy[0].decoder.superresolution['conv_sr'](latent)

        # Rearrange dimensions based on decoder config
        if not self.vae_proxy[0].decoder.D_roll_out_input:
            latent = rearrange(latent, '(b n) c h w->b (n c) h w', n=3)
        else:
            latent = rearrange(latent, 'b c h (n w)->b (n c) h w', n=3)

        return latent
    
    def unflatten_token(self, x, p=None):
        """Unflatten 1D token sequence into 2D feature maps"""
        B, L, C = x.shape
        x = x.reshape(B, 3, L // 3, C)

        # Remove classifier token if present
        if self.vae_proxy[0].decoder.cls_token:
            x = x[:, :, 1:]

        # Calculate spatial dimensions
        h = w = int((x.shape[2])**.5)
        assert h * w == x.shape[2]

        # Reshape based on patch size
        if p is None:
            x = x.reshape(shape=(B, 3, h, w, -1))
            if not self.vae_proxy[0].decoder.D_roll_out_input:
                x = rearrange(x, 'b n h w c->(b n) c h w')
            else:
                x = rearrange(x, 'b n h w c->b c h (n w)')
        else:
            x = x.reshape(shape=(B, 3, h, w, p, p, -1))
            if self.vae_proxy[0].decoder.D_roll_out_input:
                x = rearrange(x, 'b n h w p1 p2 c->b c (h p1) (n w p2)')
            else:
                x = rearrange(x, 'b n h w p1 p2 c->(b n) c (h p1) (w p2)')

        return x

    def forward(self, pooler_output, dino_condition, x_BLCv_wo_first_l, empty_pooler_output, empty_dino_embedding) -> torch.Tensor:
        """Forward pass of the VAR model
        
        Args:
            pooler_output: Pooled output features from DINO encoder
            dino_condition: DINO image embeddings for conditioning
            x_BLCv_wo_first_l: Input tokens for teacher forcing (B, L-first_l, C)
            empty_pooler_output: Empty pooler output for classifier-free guidance
            empty_dino_embedding: Empty DINO embeddings for classifier-free guidance
            
        Returns:
            torch.Tensor: Output logits of shape (B, L, vocab_size)
        """
        # Get sequence length based on progressive training stage
        ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L * 3)[1]
        B = x_BLCv_wo_first_l.shape[0]
        
        with torch.cuda.amp.autocast(enabled=False):
            # Apply classifier-free guidance by randomly replacing conditions
            replace_pooler = torch.rand(B, device=pooler_output.device) < self.cond_drop_rate
            pooler_output[replace_pooler] = empty_pooler_output
            
            replace_condition = torch.rand(B, device=pooler_output.device) < self.cond_drop_rate
            dino_condition[replace_condition] = empty_dino_embedding

            # Generate start tokens from pooler output
            sos = cond_BD = self.pooler_emb(pooler_output)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            sos = sos.repeat(1, 3, 1)
            
            # Concatenate start tokens with input sequence
            x_BLC = sos if self.prog_si == 0 else torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)

            # Add positional embeddings:
            # - Level embedding for each scale
            # - Position embedding for each token
            # - Plane embedding for each plane
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]
            x_BLC = x_BLC + self.plane_embed(self.plane_1L[:, :ed].expand(B, -1))

        # Get attention bias mask and condition embeddings
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # Handle mixed precision training
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        # Forward pass through transformer blocks
        for b in self.blocks:
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, dino_condition=dino_condition, attn_bias=attn_bias)
            
        # Get final logits
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)

        # Add regularization term if in first stage
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
                
        return x_BLC

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        """Initialize model weights
        
        Args:
            init_adaln: Scale for AdaLN weights
            init_adaln_gamma: Scale for AdaLN gamma weights  
            init_head: Scale for output head weights
            init_std: Standard deviation for weight initialization
            conv_std_or_gain: Standard deviation or gain for conv layers
        """
        # Compute initialization std if auto mode
        if init_std < 0:
            init_std = (1 / self.C / 3) ** 0.5
            
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        
        # Initialize weights for different layer types
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
                
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
                
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                              nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
                
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()

        # Initialize output head
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()

        # Initialize AdaLN layers
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()

        # Initialize transformer blocks
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
                
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

    def extra_repr(self):
        """Extra string representation"""
        return f'drop_path_rate={self.drop_path_rate:g}'


class VAR_text(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        """Initialize VAR_text model
        
        Args:
            vae_local (VQVAE): Local VQVAE model
            num_classes (int): Number of classes
            depth (int): Number of transformer layers
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            mlp_ratio (float): MLP hidden dim expansion ratio
            drop_rate (float): Dropout rate
            attn_drop_rate (float): Attention dropout rate
            drop_path_rate (float): Drop path rate
            norm_eps (float): Layer norm epsilon
            shared_aln (bool): Whether to share AdaLN layers
            cond_drop_rate (float): Condition dropout rate for classifier-free guidance
            attn_l2_norm (bool): Whether to use L2 norm for attention
            patch_nums (tuple): Progressive patch numbers for each stage
            flash_if_available (bool): Whether to use flash attention
            fused_if_available (bool): Whether to use fused operations
        """
        super().__init__()
        # Model hyperparameters
        assert embed_dim % num_heads == 0
        
        self.Cvae, self.V = vae_local.decoder.superresolution.quantize.Cvae, vae_local.decoder.superresolution.quantize.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # Progressive training index
        
        # Progressive patch configuration
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # Input (word) embedding
        quant: VectorQuantizer2 = vae_local.decoder.superresolution.quantize

        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # Class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        
        # Change input pooler output dimension to match class embedding (768 -> 1024)
        self.pooler_emb = nn.Linear(768, self.C)
        nn.init.trunc_normal_(self.pooler_emb.weight.data, mean=0, std=init_std)
        nn.init.trunc_normal_(self.pooler_emb.bias.data, mean=0, std=init_std)
        
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # Absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn*3, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L*3, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        
        # Level embedding (similar to GPT's segment embedding)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # Plane embedding
        self.plane_embed = nn.Embedding(3, self.C)
        nn.init.trunc_normal_(self.plane_embed.weight.data, mean=0, std=init_std)
        
        # Backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # Stochastic depth decay rule
        
        self.blocks = nn.ModuleList([
            AdaLNCrossSelfAttn_text(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # Attention mask for training (masking future tokens)
        d: torch.Tensor = torch.cat([torch.full((pn*pn*3,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L * 3, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L * 3, self.L * 3)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # Plane embedding buffer
        p = []
        for pn in self.patch_nums:
            p.append(torch.tensor([0]).repeat(pn*pn))
            p.append(torch.tensor([1]).repeat(pn*pn))
            p.append(torch.tensor([2]).repeat(pn*pn))
        p_exp = torch.cat(p).unsqueeze(0)
        plane_1L = p_exp.contiguous()
        self.register_buffer('plane_1L', plane_1L)

        # Classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
 


    @torch.no_grad()
    def autoregressive_infer_cfg_3D_VAR_text_l2norm(
        self, B: int,
        dino_image_embeddings: Optional[Union[int, torch.LongTensor]], 
        pooler_output: Optional[int] = None,
        cfg=1.5, top_k=0, top_p=0.0,
        g_seed: Optional[int] = None,
        more_smooth=False,
    ) -> torch.Tensor:
        """
        Autoregressive inference with classifier-free guidance for text-to-3D generation.
        
        Args:
            B: Batch size
            dino_image_embeddings: Text embeddings
            pooler_output: Pooled text features
            cfg: Classifier-free guidance ratio
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            g_seed: Random seed
            more_smooth: Whether to use Gumbel softmax for smoother outputs
            
        Returns:
            Tuple of:
            - Generated triplane latents
            - Generated token indices
        """
        # Setup RNG if seed provided
        rng = None if g_seed is None else self.rng.manual_seed(g_seed); rng = self.rng

        # Load empty embeddings for classifier-free guidance
        empty_pooler = torch.from_numpy(np.load("./files/empty_text_pooler_output.npy")).to(pooler_output.device).unsqueeze(0)
        empty_dino = torch.from_numpy(np.load("./files/empty_text_embedding.npy")).to(pooler_output.device).unsqueeze(0)
        
        # Concatenate real and empty embeddings for CFG
        pooler_output = torch.cat((pooler_output, empty_pooler.expand(pooler_output.shape)), dim=0)
        dino_image_embeddings = torch.cat((dino_image_embeddings, empty_dino.expand(dino_image_embeddings.shape)), dim=0)

        # Get initial tokens and positional embeddings
        sos = cond_BD = self.pooler_emb(pooler_output)
        sos = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1)
        sos = sos.repeat(1, 3, 1)
        
        # Add level and plane embeddings
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC + self.plane_embed(self.plane_1L)
        next_token_map = sos.expand(2 * B, 3, -1) + lvl_pos[:, :3]
        
        # Initialize feature maps for each plane
        cur_L = 0
        f_hat_list = [sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]) for _ in range(3)]

        # Enable KV caching for transformer blocks
        for b in self.blocks:
            b.attn.kv_caching(True)

        # Generate tokens autoregressively
        for si, pn in enumerate(self.patch_nums):
            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn*3

            # Forward pass through transformer
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, dino_condition=dino_image_embeddings, attn_bias=None)
            
            # Get logits and apply classifier-free guidance
            logits_BlV = self.get_logits(x, cond_BD)
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            # Sample next tokens
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if si == 0:
                g_BL = idx_Bl
            else:
                g_BL = torch.cat((g_BL, idx_Bl), dim=1)

            # Get embeddings for sampled tokens
            if not more_smooth:
                embedding = F.normalize(self.vae_quant_proxy[0].embedding.weight, p=2, dim=-1)
                h_BChw_concate = embedding[idx_Bl]
            else:
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                embedding = F.normalize(self.vae_quant_proxy[0].embedding.weight, p=2, dim=-1)
                h_BChw_concate = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ embedding.unsqueeze(0)

            # Process embeddings for each plane
            h_BChw_list = h_BChw_concate.split(int(h_BChw_concate.shape[1]/3), dim=1)
            next_token_map_list = []
            for i in range(3):
                h_BChw = h_BChw_list[i].transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                f_hat_list[i], next_token_map_single_plane = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_list[i], h_BChw)
                next_token_map_list.append(next_token_map_single_plane.view(B, self.Cvae, -1).transpose(1, 2))

            # Prepare next token map
            if si != self.num_stages_minus_1:
                next_token_map = torch.cat(next_token_map_list, dim=1)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + 3 * self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)

        # Disable KV caching
        for b in self.blocks:
            b.attn.kv_caching(False)

        # Combine feature maps from all planes
        f_hat_all = torch.zeros((3 * B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]), device=f_hat_list[0].device, dtype=f_hat_list[0].dtype)
        f_hat_all[0::3] = f_hat_list[0]
        f_hat_all[1::3] = f_hat_list[1]
        f_hat_all[2::3] = f_hat_list[2]

        # Decode feature maps to triplane representation
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            f_hat_all = self.vae_proxy[0].decoder.superresolution['post_quant_conv'](f_hat_all.to(torch.bfloat16))
            f_hat_all = f_hat_all.reshape(f_hat_all.shape[0] // 3, -1, f_hat_all.shape[-2], f_hat_all.shape[-1])
            f_hat_all = self.vae_proxy[0].decoder.superresolution['ldm_upsample'](f_hat_all)
            f_hat_all = self.vae_proxy[0].decoder.forward_vit_decoder(f_hat_all, 224)
            latent_after_vit = self.vit_decode_postprocess(f_hat_all)
            return latent_after_vit, g_BL
        
    @torch.no_grad()
    def reconstruct_gt_Bl_idx(
            self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
            g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
            more_smooth=False, gt_idx: Optional[torch.LongTensor] = None,
        ) -> torch.Tensor:
        """Reconstruct ground truth tokens into image.
        
        Args:
            B: Batch size
            label_B: Class labels (unused)
            g_seed: Random seed
            cfg: Classifier-free guidance ratio (unused)
            top_k: Top-k sampling parameter (unused)
            top_p: Top-p sampling parameter (unused)
            more_smooth: Whether to use Gumbel softmax smoothing (unused)
            gt_idx: Ground truth token indices
            
        Returns:
            Reconstructed image tensor
        """
        if g_seed is not None:
            self.rng.manual_seed(g_seed)
            rng = self.rng
            
        cur_L = 0
        f_hat_list = [
            torch.zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1], device=gt_idx.device),
            torch.zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1], device=gt_idx.device),
            torch.zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1], device=gt_idx.device)
        ]
        
        # Enable KV caching
        for b in self.blocks:
            b.attn.kv_caching(True)
            
        # Progressive reconstruction
        for si, pn in enumerate(self.patch_nums):
            cur_L += pn * pn * 3
            idx_Bl = gt_idx[:, cur_L-pn*pn*3:cur_L]
            
            # Get embeddings
            embedding = F.normalize(self.vae_quant_proxy[0].embedding.weight, p=2, dim=-1)
            h_BChw_concate = embedding[idx_Bl]
            
            # Process each plane
            h_BChw_list = h_BChw_concate.split(int(h_BChw_concate.shape[1]/3), dim=1)
            for i in range(3):
                h_BChw = h_BChw_list[i].transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                f_hat_list[i], _ = self.vae_quant_proxy[0].get_next_autoregressive_input(
                    si, len(self.patch_nums), f_hat_list[i], h_BChw)
                    
        # Disable KV caching
        for b in self.blocks:
            b.attn.kv_caching(False)
            
        # Combine feature maps
        f_hat_all = torch.zeros(
            (3 * B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]),
            device=f_hat_list[0].device,
            dtype=f_hat_list[0].dtype
        )
        f_hat_all[0::3] = f_hat_list[0]
        f_hat_all[1::3] = f_hat_list[1]
        f_hat_all[2::3] = f_hat_list[2]

        # Decode feature maps
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            f_hat_all = self.vae_proxy[0].decoder.superresolution['post_quant_conv'](f_hat_all.to(torch.bfloat16))
            f_hat_all = f_hat_all.reshape(f_hat_all.shape[0] // 3, -1, f_hat_all.shape[-2], f_hat_all.shape[-1])
            f_hat_all = self.vae_proxy[0].decoder.superresolution['ldm_upsample'](f_hat_all)
            f_hat_all = self.vae_proxy[0].decoder.forward_vit_decoder(f_hat_all, 224)
            latent_after_vit = self.vit_decode_postprocess(f_hat_all)
            return latent_after_vit
        
    def vit_decode_postprocess(self, latent_from_vit):
        """Post-process ViT decoder output by unflattening tokens and applying super-resolution.
        
        Args:
            latent_from_vit: Tensor of shape (B, L, C) from ViT decoder
            
        Returns:
            Processed latents with shape (B, 3C, H, W) or (B*3, C, H, W)
        """
        # Handle optional class token
        if self.vae_proxy[0].decoder.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # Unflatten tokens into spatial dimensions
        latent = self.unflatten_token(latent_from_vit)  # B 3 H W C

        # Apply super-resolution convolution
        latent = self.vae_proxy[0].decoder.superresolution['conv_sr'](latent)

        # Rearrange dimensions based on decoder config
        if not self.vae_proxy[0].decoder.D_roll_out_input:
            latent = rearrange(latent, '(b n) c h w->b (n c) h w', n=3)
        else:
            latent = rearrange(latent, 'b c h (n w)->b (n c) h w', n=3)

        return latent
    
    def unflatten_token(self, x, p=None):
        """Unflatten sequence of tokens into spatial dimensions.
        
        Args:
            x: Input tensor of shape (B, L, C)
            p: Optional patch size for hierarchical unflattening
            
        Returns:
            Unflattened tensor with spatial dimensions
        """
        B, L, C = x.shape
        x = x.reshape(B, 3, L // 3, C)

        if self.vae_proxy[0].decoder.cls_token:
            x = x[:, :, 1:]  # Remove CLS token

        h = w = int((x.shape[2])**.5)
        assert h * w == x.shape[2]

        if p is None:
            x = x.reshape(shape=(B, 3, h, w, -1))
            if not self.vae_proxy[0].decoder.D_roll_out_input:
                x = rearrange(x, 'b n h w c->(b n) c h w')
            else:
                x = rearrange(x, 'b n h w c->b c h (n w)')
        else:
            x = x.reshape(shape=(B, 3, h, w, p, p, -1))
            if self.vae_proxy[0].decoder.D_roll_out_input:
                x = rearrange(x, 'b n h w p1 p2 c->b c (h p1) (n w p2)')
            else:
                x = rearrange(x, 'b n h w p1 p2 c->(b n) c (h p1) (w p2)')

        return x

 
    def forward(self, pooler_output, dino_condition, x_BLCv_wo_first_l, empty_pooler_output, empty_dino_embedding) -> torch.Tensor:
        """Forward pass of the VAR model.
        
        Args:
            pooler_output: Pooled text features
            dino_condition: DINO image embeddings
            x_BLCv_wo_first_l: Teacher forcing input (B, L-first_l, C_vae)
            empty_pooler_output: Empty pooler output for classifier-free guidance
            empty_dino_embedding: Empty DINO embeddings for classifier-free guidance
            
        Returns:
            Logits tensor of shape (B, L, V) where V is vocab size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L * 3)
        B = x_BLCv_wo_first_l.shape[0]

        with torch.cuda.amp.autocast(enabled=False):
            # Apply classifier-free guidance dropout
            replace_pooler = torch.rand(B, device=pooler_output.device) < self.cond_drop_rate
            pooler_output[replace_pooler] = empty_pooler_output
            
            replace_condition = torch.rand(B, device=pooler_output.device) < self.cond_drop_rate
            dino_condition[replace_condition] = empty_dino_embedding

            # Generate start tokens from pooler output
            sos = cond_BD = self.pooler_emb(pooler_output)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            sos = sos.repeat(1, 3, 1)

            # Combine start tokens with input sequence
            x_BLC = sos if self.prog_si == 0 else torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)

            # Add positional embeddings
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]
            x_BLC = x_BLC + self.plane_embed(self.plane_1L[:, :ed].expand(B, -1))

        # Get attention bias mask
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # Handle mixed precision
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        # Forward through transformer blocks
        for block in self.blocks:
            x_BLC = block(x=x_BLC, cond_BD=cond_BD_or_gss, dino_condition=dino_condition, attn_bias=attn_bias)
        
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)

        # Add gradient path for word embeddings
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s

        return x_BLC
    
    
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        """Initialize model weights.
        
        Args:
            init_adaln: AdaLN initialization scale
            init_adaln_gamma: AdaLN gamma initialization scale  
            init_head: Head layer initialization scale
            init_std: Standard deviation for weight initialization
            conv_std_or_gain: Conv layer initialization scale/gain
        """
        if init_std < 0:
            init_std = (1 / self.C / 3) ** 0.5

        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        
        # Initialize standard layers
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias:
                    m.bias.data.zero_()
                    
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
                    
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                              nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight:
                    m.weight.data.fill_(1.)
                if with_bias:
                    m.bias.data.zero_()
                    
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias:
                    m.bias.data.zero_()

        # Initialize head
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()

        # Initialize AdaLN head
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()

        # Initialize transformer blocks
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
                
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        """Extra string representation."""
        return f'drop_path_rate={self.drop_path_rate:g}'

class VARHF(VAR, PyTorchModelHubMixin):
    """VAR model with HuggingFace model hub integration"""
    
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000,
        depth=16, 
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4.,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 progressive steps
        flash_if_available=True,
        fused_if_available=True,
    ):
        """Initialize VARHF model
        
        Args:
            vae_kwargs: Arguments for VAE initialization
            num_classes: Number of classes for conditioning
            depth: Number of transformer layers
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim expansion ratio
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Drop path rate
            norm_eps: Layer norm epsilon
            shared_aln: Whether to share AdaLN layers
            cond_drop_rate: Condition dropout rate for classifier-free guidance
            attn_l2_norm: Whether to use L2 norm for attention
            patch_nums: Progressive patch numbers for each stage
            flash_if_available: Whether to use flash attention
            fused_if_available: Whether to use fused operations
        """
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes,
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_eps=norm_eps,
            shared_aln=shared_aln,
            cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available,
            fused_if_available=fused_if_available,
        )
