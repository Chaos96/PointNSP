import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.helpers import DropPath, drop_path
from ldm.modules.x_transformer import RMSNorm
from ipdb import set_trace as st

# This file provides 3 core blocks used in VAR transformer
__all__ = ['FFN', 'AdaLNSelfAttn', 'AdaLNBeforeHead']

# Try to import optimized operators for better performance
# 1. Fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError: pass

# 2. Faster attention implementations
try: from xformers.ops import memory_efficient_attention  
except ImportError: pass
try: from flash_attn import flash_attn_func  # qkv: BLHc, ret: BLHcq
except ImportError: pass
try: from torch.nn.functional import scaled_dot_product_attention as slow_attn  # q,k,v: BHLc
except ImportError:
    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        # Compute attention scores
        attn = query.mul(scale) @ key.transpose(-2, -1)  # BHLc @ BHcL => BHLL
        if attn_mask is not None: attn.add_(attn_mask)
        
        # Apply softmax and dropout
        attn = F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)
        
        # Compute weighted sum of values
        return attn @ value

# Feed-forward network block
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., fused_if_available=True):
        super().__init__()
        # Use fused MLP if available for better performance
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        
        # Configure dimensions
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # Build network
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()
    
    def forward(self, x):
        if self.fused_mlp_func is not None:
            # Use fused implementation
            return self.drop(self.fused_mlp_func(
                x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, 
                bias1=self.fc1.bias, bias2=self.fc2.bias,
                activation='gelu_approx', save_pre_act=self.training, 
                return_residual=False, checkpoint_lvl=0,
                heuristic=0, process_group=None,
            ))
        else:
            # Standard implementation
            return self.drop(self.fc2(self.act(self.fc1(x))))
    
    def extra_repr(self) -> str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}'

# Self-attention block
class SelfAttention(nn.Module):
    def __init__(
        self, block_idx, embed_dim=768, num_heads=12,
        attn_drop=0., proj_drop=0., attn_l2_norm=False, flash_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        # Basic attention parameters
        self.block_idx = block_idx
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_l2_norm = attn_l2_norm
        
        # Configure attention scaling
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        
        # QKV projection
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(embed_dim))
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        
        # Attention dropout
        self.attn_drop = attn_drop
        
        # Use optimized attention implementations if available
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None
        
        # KV caching for inference
        self.caching = False
        self.cached_k = None 
        self.cached_v = None
    
    def kv_caching(self, enable: bool):
        self.caching = enable
        self.cached_k = None
        self.cached_v = None
    
    def forward(self, x, attn_bias):
        B, L, C = x.shape
        
        # Project input to Q, K, V
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, 
                      bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype
        
        # Prepare Q, K, V based on attention implementation
        using_flash = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        if using_flash or self.using_xform:
            q, k, v = qkv.unbind(dim=2)  # Shape: BLHc
            dim_cat = 1
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # Shape: BHLc
            dim_cat = 2
        
        # Apply L2 normalization if enabled
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform:
                scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        
        # Handle KV caching during inference
        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
        
        # Compute attention with the appropriate implementation
        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            # Use Flash Attention
            oup = flash_attn_func(q.to(dtype=main_type), k.to(dtype=main_type), 
                                v.to(dtype=main_type), dropout_p=dropout_p, 
                                softmax_scale=self.scale).view(B, L, C)
        elif self.using_xform:
            # Use xFormers memory efficient attention
            attn_bias_expanded = None if attn_bias is None else attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1)
            oup = memory_efficient_attention(q.to(dtype=main_type), k.to(dtype=main_type),
                                          v.to(dtype=main_type), attn_bias=attn_bias_expanded,
                                          p=dropout_p, scale=self.scale).view(B, L, C)
        else:
            # Use standard attention
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale,
                          attn_mask=attn_bias, dropout_p=dropout_p)
            oup = oup.transpose(1, 2).reshape(B, L, C)
        
        # Project output
        return self.proj_drop(self.proj(oup))
    
    def extra_repr(self) -> str:
        return f'using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}'

class CrossAttention(nn.Module):
    def __init__(
        self, block_idx, query_dim=1024, key_value_dim=768, num_heads=12,
        attn_drop=0., proj_drop=0., attn_l2_norm=False, flash_if_available=True,
    ):
        super().__init__()
        assert query_dim % num_heads == 0
        assert key_value_dim % num_heads == 0
        self.block_idx, self.num_heads = block_idx, num_heads
        self.query_head_dim = query_dim // num_heads  # head dimension for query
        self.key_value_head_dim = key_value_dim // num_heads  # head dimension for key and value
        self.attn_l2_norm = attn_l2_norm

        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.key_value_head_dim)
        
        self.mat_q = nn.Linear(query_dim, query_dim, bias=False)
        self.mat_kv = nn.Linear(key_value_dim, query_dim * 2, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(query_dim))
        self.v_bias = nn.Parameter(torch.zeros(query_dim))
        self.register_buffer('zero_k_bias', torch.zeros(query_dim))
        
        self.proj = nn.Linear(query_dim, query_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None
        
        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None
    
    def kv_caching(self, enable: bool): 
        self.caching, self.cached_k, self.cached_v = enable, None, None
    
    def forward(self, x_q, x_kv, attn_bias=None):
        B, L_q, C_q = x_q.shape  # x_q: (Batch, Query Length, Query Dim)
        B, L_kv, C_kv = x_kv.shape  # x_kv: (Batch, Key/Value Length, Key/Value Dim)
        
        # Compute Q, K, V
        q = F.linear(input=x_q, weight=self.mat_q.weight, bias=self.q_bias).view(B, L_q, self.num_heads, self.query_head_dim)
        kv = F.linear(input=x_kv, weight=self.mat_kv.weight, bias=torch.cat((self.zero_k_bias, self.v_bias))).view(B, L_kv, 2, self.num_heads, self.query_head_dim)
        assert q.dtype == kv.dtype
        main_type = q.dtype

        using_flash = self.using_flash and attn_bias is None and q.dtype != torch.float32 and kv.dtype != torch.float32
        if using_flash or self.using_xform:
            k, v = kv.unbind(dim=2)
            dim_cat = 1
        else:
            k, v = kv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            q = q.permute(0, 2, 1, 3)
            dim_cat = 2

        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if self.using_flash or self.using_xform:
                scale_mul = scale_mul.transpose(1, 2)
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        
        if self.caching:
            if self.cached_k is None: 
                self.cached_k = k
                self.cached_v = v
            else: 
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)

        dropout_p = self.attn_drop if self.training else 0.0
        
        if using_flash:
            oup = flash_attn_func(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), 
                                dropout_p=dropout_p, softmax_scale=self.scale).view(B, L_q, C_q)
        elif self.using_xform:
            attn_bias_expanded = None if attn_bias is None else attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1)
            oup = memory_efficient_attention(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type),
                                          attn_bias=attn_bias_expanded, p=dropout_p, scale=self.scale).view(B, L_q, C_q)
        else:
            raise NotImplementedError

        return self.proj_drop(self.proj(oup))

class AdaLNSelfAttn(nn.Module):
    def __init__(
        self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: bool, norm_layer,
        num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
        flash_if_available=False, fused_if_available=True,
    ):
        super(AdaLNSelfAttn, self).__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = SelfAttention(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available)
        self.ffn = FFN(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=drop, fused_if_available=fused_if_available)
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)
        
        self.fused_add_norm_fn = None
    
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, cond_BD, attn_bias):   # C: embed_dim, D: cond_dim
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        x = x + self.drop_path(self.attn( self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1), attn_bias=attn_bias ).mul_(gamma1))
        x = x + self.drop_path(self.ffn( self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed when FusedMLP is used
        return x
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}'

class AdaLNCrossSelfAttn_Image_new(nn.Module):
    """
    Adaptive Layer Norm Cross Self Attention module for image features.
    Combines cross attention with DINO features and self attention with adaptive layer norm.
    """
    def __init__(
        self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: bool, norm_layer,
        num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
        flash_if_available=False, fused_if_available=True,
    ):
        super().__init__()
        # Basic parameters
        self.block_idx = block_idx
        self.last_drop_p = last_drop_p
        self.C = embed_dim  # Embedding dimension
        self.D = cond_dim   # Condition dimension
        
        # Dropout path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Attention modules
        self.attn = SelfAttention(
            block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop, attn_l2_norm=attn_l2_norm,
            flash_if_available=flash_if_available
        )
        self.cross_attn = CrossAttention(
            block_idx=block_idx, query_dim=embed_dim, key_value_dim=1024,
            num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop,
            attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available
        )
        
        # Feed forward network
        self.ffn = FFN(
            in_features=embed_dim,
            hidden_features=round(embed_dim * mlp_ratio),
            drop=drop,
            fused_if_available=fused_if_available
        )
        
        # Normalization layers
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.prenorm_ca_dino = RMSNorm(embed_dim, eps=1e-5)
        
        # Adaptive layer norm parameters
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            self.ada_lin = nn.Sequential(
                nn.SiLU(inplace=False),
                nn.Linear(cond_dim, 6*embed_dim)
            )
        
        self.fused_add_norm_fn = None

    def forward(self, x, cond_BD, dino_condition, attn_bias):
        """
        Forward pass through the module.
        
        Args:
            x: Input tensor
            cond_BD: Conditioning tensor
            dino_condition: DINO feature tensor
            attn_bias: Attention bias tensor
            
        Returns:
            Processed tensor after cross attention, self attention and FFN
        """
        # Get adaptive layer norm parameters
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2)
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)

        # Cross attention with DINO features
        x = x + self.drop_path(self.cross_attn(self.prenorm_ca_dino(x), dino_condition))
        
        # Self attention with adaptive layer norm
        x = x + self.drop_path(
            self.attn(
                self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1),
                attn_bias=attn_bias
            ).mul_(gamma1)
        )
        
        # Feed forward with adaptive layer norm
        x = x + self.drop_path(
            self.ffn(
                self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)
            ).mul(gamma2)
        )
        
        return x
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}'

    
class AdaLNCrossSelfAttn_text(nn.Module):
    """
    Adaptive Layer Norm Cross Self Attention module for text conditioning.
    
    Args:
        block_idx: Index of transformer block
        last_drop_p: Last dropout probability
        embed_dim: Embedding dimension
        cond_dim: Conditioning dimension
        shared_aln: Whether to share adaptive layer norm
        norm_layer: Normalization layer
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Drop path rate
        attn_l2_norm: Whether to use L2 norm for attention
        flash_if_available: Whether to use flash attention
        fused_if_available: Whether to use fused operations
    """
    def __init__(
        self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: bool, norm_layer,
        num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
        flash_if_available=False, fused_if_available=True,
    ):
        super(AdaLNCrossSelfAttn_text, self).__init__()
        self.block_idx = block_idx
        self.last_drop_p = last_drop_p
        self.C = embed_dim
        self.D = cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Initialize attention layers
        self.attn = SelfAttention(
            block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop, attn_l2_norm=attn_l2_norm,
            flash_if_available=flash_if_available
        )
        
        self.ffn = FFN(
            in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio),
            drop=drop, fused_if_available=fused_if_available
        )
        
        self.cross_attn = CrossAttention(
            block_idx=block_idx, query_dim=embed_dim, key_value_dim=768,
            num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop,
            attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available
        )
        
        # Initialize normalization layers
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        self.prenorm_ca_dino = RMSNorm(embed_dim, eps=1e-5)
        
        # Initialize adaptive layer norm parameters
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            self.ada_lin = nn.Sequential(
                nn.SiLU(inplace=False),
                nn.Linear(cond_dim, 6*embed_dim)
            )
        
        self.fused_add_norm_fn = None

    def forward(self, x, cond_BD, dino_condition, attn_bias):
        """
        Forward pass through the module.
        
        Args:
            x: Input tensor
            cond_BD: Conditioning tensor
            dino_condition: DINO feature tensor
            attn_bias: Attention bias tensor
            
        Returns:
            Processed tensor after attention and FFN layers
        """
        # Get adaptive layer norm parameters
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2)
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)

        # Self attention with adaptive layer norm
        x = x + self.drop_path(
            self.attn(
                self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1),
                attn_bias=attn_bias
            ).mul_(gamma1)
        )
        
        # Cross attention with DINO features
        x = x + self.drop_path(self.cross_attn(self.prenorm_ca_dino(x), dino_condition))
        
        # Feed forward with adaptive layer norm
        x = x + self.drop_path(
            self.ffn(
                self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)
            ).mul(gamma2)
        )
        
        return x
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}'

class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2*C))
    
    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)
