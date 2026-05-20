"""
Model initialization and configuration module.
Contains functions to build VAE and VAR models for 3D generation.
"""

from typing import Tuple
import torch.nn as nn
import os
import sys

# Import model components
from vit.quant import VectorQuantizer2 
from .var import VAR, VAR_text
from .vqvae import VQVAE
from .model_config import encoder_and_nsr_defaults

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from nsr.script_util import create_3DAE_model, create_3DAE_model_mesh



def build_vae_var_3D_VAR(
    device,
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # Progressive training steps
    num_classes=1000,    # Number of classes
    depth=16,            # Transformer depth
    shared_aln=False,    # Share AdaLN layers
    attn_l2_norm=True,   # L2 normalize attention
    flash_if_available=True,     # Use flash attention if available
    fused_if_available=True,     # Use fused operations if available
    init_adaln=0.5,             # AdaLN init value
    init_adaln_gamma=1e-5,      # AdaLN gamma init
    init_head=0.02,             # Head layer init
    init_std=-1,                # Weight init std (-1 for auto)
    args=None                   # Additional config args
) -> Tuple[VQVAE, VAR]:
    """
    Build VAE and VAR models for 3D generation.
    
    Args:
        device: Device to place models on
        patch_nums: Progressive training patch sizes
        num_classes: Number of output classes
        depth: Transformer depth
        attn_l2_norm: Attention parameters
        flash_if_available, fused_if_available: Optimization flags
        init_*: Initialization parameters
        args: Additional configuration arguments
        
    Returns:
        vae_local: VQVAE model
        var_wo_ddp: VAR model
    """
    # Model architecture parameters
    heads = 16
    width = depth * 64
    dpr = 0.1 * depth/24

    # Disable default initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, 
               nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)

    # Build VQVAE model
    args.LN3DiffConfig.img_size = [args.LN3DiffConfig.image_size_encoder]
    
    if not args.flexicubes:
        vae_local = create_3DAE_model(**args_to_dict(args.LN3DiffConfig,
                       encoder_and_nsr_defaults().keys())).to(device)
    else:
        vae_local = create_3DAE_model_mesh(**args_to_dict(args.LN3DiffConfig,
                       encoder_and_nsr_defaults().keys())).to(device)
        vae_local.decoder.triplane_decoder.init_flexicubes_geometry(device=device, fovy=43)

    # Build VAR model
    if args.text_conditioned:
        var_wo_ddp = VAR_text(
            vae_local=vae_local,
            num_classes=num_classes, 
            depth=depth,
            embed_dim=width,
            num_heads=heads,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=dpr,
            norm_eps=1e-6,
            shared_aln=shared_aln,
            cond_drop_rate=0.1,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available,
            fused_if_available=fused_if_available,
        ).to(device) 
    else: 
        var_wo_ddp = VAR(
            vae_local=vae_local,
            num_classes=num_classes, 
            depth=depth,
            embed_dim=width,
            num_heads=heads,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=dpr,
            norm_eps=1e-6,
            shared_aln=shared_aln,
            cond_drop_rate=0.1,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available,
            fused_if_available=fused_if_available,
        ).to(device)
    
    var_wo_ddp.init_weights(init_adaln=init_adaln, 
                           init_adaln_gamma=init_adaln_gamma,
                           init_head=init_head,
                           init_std=init_std)

    return vae_local, var_wo_ddp


def args_to_dict(args, keys):
    """Convert args to dictionary with specified keys"""
    for k in keys:
        if not hasattr(args, k):
            print(f"Missing key {k} in args")
    return {k: getattr(args, k) for k in keys}