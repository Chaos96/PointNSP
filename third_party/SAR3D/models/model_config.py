# Model configuration defaults
def encoder_and_nsr_defaults():
    """Default configuration for image encoder"""
    res = dict(
        dino_version='v1',
        encoder_in_channels=3,
        img_size=[224],
        patch_size=16,  # ViT-S/16
        in_chans=384,
        num_classes=0,
        embed_dim=384,  # ViT encoder dimension
        depth=6,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer='nn.LayerNorm',
        cls_token=False,
        encoder_cls_token=False,
        decoder_cls_token=False,
        sr_kwargs={},
        sr_ratio=2,
    )
    res.update(model_encoder_defaults())
    res.update(nsr_decoder_defaults())
    res.update(ae_classname='vit.vit_triplane.ViTTriplaneDecomposed')
    return res


def model_encoder_defaults():
    """Default configuration for model encoder"""
    return dict(
        use_clip=False,
        arch_encoder="vits",
        arch_decoder="vits",
        load_pretrain_encoder=False,
        encoder_lr=1e-5,
        encoder_weight_decay=0.001,
        no_dim_up_mlp=False,
        dim_up_mlp_as_func=False,
        decoder_load_pretrained=True,
        uvit_skip_encoder=False,
        vae_p=1,
        ldm_z_channels=4,
        ldm_embed_dim=4,
        use_conf_map=False,
        sd_E_ch=64,
        z_channels=3*4,
        sd_E_num_res_blocks=1,
        arch_dit_decoder='DiT2-B/2',
        return_all_dit_layers=False,
        lrm_decoder=False,
        gs_rendering=False,
    )


def nsr_decoder_defaults():
    """Default configuration for NSR decoder"""
    res = {'decomposed': False}
    res.update(triplane_decoder_defaults())
    res.update(vit_decoder_defaults())
    return res


def triplane_decoder_defaults():
    """Default configuration for triplane decoder"""
    return dict(
        triplane_fg_bg=False,
        cfg='shapenet',
        density_reg=0.25,
        density_reg_p_dist=0.004,
        reg_type='l1',
        triplane_decoder_lr=0.0025,
        super_resolution_lr=0.0025,
        c_scale=1,
        nsr_lr=0.02,
        triplane_size=224,
        decoder_in_chans=32,
        triplane_in_chans=-1,
        decoder_output_dim=3,
        out_chans=96,
        c_dim=25,
        ray_start=0.6,
        ray_end=1.8,
        rendering_kwargs={},
        sr_training=False,
        bcg_synthesis=False,
        bcg_synthesis_kwargs={},
        image_size=128,
        patch_rendering_resolution=45,
    )


def vit_decoder_defaults():
    """Default configuration for ViT decoder"""
    return dict(
        vit_decoder_lr=1e-5,
        vit_decoder_wd=0.001,
    )