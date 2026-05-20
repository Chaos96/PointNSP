#    Copyright 2023 Runsen Xu

from typing import Sequence, Optional, Union, Tuple, List

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from .utils import *
from pointllm.utils import *

from contextlib import nullcontext
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

import os
from ipdb import set_trace as st
from .ldm.modules.diffusionmodules.mv_unet import MVUNet, LGM_MVEncoder
from .ldm.modules.diffusionmodules.model import Encoder, MVEncoder, MVEncoderGS
from .ldm.modules.diffusionmodules.mv_unet import MVUNet, LGM_MVEncoder




from .ldm.modules.diffusionmodules.model import Encoder, Decoder, MVEncoder, MVEncoderGS, MVEncoderGSDynamicInp
from .vit import vision_transformer as vits
from .vit.vit_triplane import Triplane, ViTTriplaneDecomposed
from ..data.nsr.triplane import TriplaneMesh
from ..data.guided_diffusion import dist_util

# * add logger
import logging
logger = logging.getLogger(__name__)

class PointLLMConfig(LlamaConfig):
    model_type = "pointllm"



class PointLLMLlamaModel(LlamaModel):
    config_class = PointLLMConfig 

    def __init__(self, config: LlamaConfig):
        super(PointLLMLlamaModel, self).__init__(config)

        self.point_backbone_type = config.point_backbone
        logger.info(f"Using {self.point_backbone_type}.")

        if self.point_backbone_type == "PointBERT":
            from pointllm.model import PointTransformer
            # address of config file, in the same dir of this file
            point_bert_config_name = getattr(config, "point_backbone_config_name", "PointTransformer_8192point_2layer") # * default for v1.2, v1.1 uses PointTransformer_base_8192point.yaml
            point_bert_config_addr = os.path.join(os.path.dirname(__file__), "pointbert", f"{point_bert_config_name}.yaml")
            print(f"Loading PointBERT config from {point_bert_config_addr}.")
            point_bert_config = cfg_from_yaml_file(point_bert_config_addr)
            if getattr(config, "use_color", False):
                point_bert_config.model.point_dims = 6
            use_max_pool = getattr(point_bert_config.model, "use_max_pool", False) # * default is false
            
            self.point_backbone = PointTransformer(point_bert_config.model, use_max_pool=use_max_pool)
            logger.info(f"Using {self.point_backbone.point_dims} dim of points.")

            self.point_backbone_config = {
                "point_cloud_dim": point_bert_config.model.point_dims,
                "backbone_output_dim": point_bert_config.model.trans_dim if not use_max_pool else point_bert_config.model.trans_dim * 2,
                "project_output_dim": self.config.hidden_size,
                "point_token_len": point_bert_config.model.num_group + 1 if not use_max_pool else 1, # * number of output features, with cls token
                "mm_use_point_start_end": self.config.mm_use_point_start_end,
                "projection_hidden_layer": point_bert_config.model.get('projection_hidden_layer', 0),
                "use_max_pool": use_max_pool
            }
            if point_bert_config.model.get('projection_hidden_layer', 0) > 0:
                self.point_backbone_config["projection_hidden_dim"] = point_bert_config.model.projection_hidden_dim # a list
            
            logger.info(f"Use max pool is {use_max_pool}. Number of point token is {self.point_backbone_config['point_token_len']}.")

        # * print relevant info with projection layers
        backbone_output_dim = self.point_backbone_config["backbone_output_dim"]
        logger.info(f"Point backbone output dim: {backbone_output_dim}.")
        logger.info(f"Use {self.point_backbone_config['projection_hidden_layer']} projection hiddent layers.")
        if self.point_backbone_config['projection_hidden_layer'] > 0:
            # Add projection layer with linear layers and GELU activation
            projection_layers = []
            last_dim = backbone_output_dim
            for i in range(point_bert_config.model.projection_hidden_layer):
                projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["projection_hidden_dim"][i]))
                projection_layers.append(nn.GELU())
                last_dim = self.point_backbone_config["projection_hidden_dim"][i]

            projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["project_output_dim"]))
            self.point_proj = nn.Sequential(*projection_layers)
            logger.info(f"Each layer with {point_bert_config.model.projection_hidden_dim} hidden units.")
        else:
            # Single layer
            self.point_proj = nn.Linear(backbone_output_dim, self.point_backbone_config['project_output_dim'])
        logger.info(f"Point projector output dim: {self.point_backbone_config['project_output_dim']}.")

        self.fix_pointnet = False
        self.fix_llm = False

    def load_point_backbone_checkpoint(self, checkpoint_path=None):
        self.point_backbone.load_checkpoint(self.config.point_backbone_ckpt if checkpoint_path is None else checkpoint_path)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        st()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        point_backbone = getattr(self, 'point_backbone', None)
        point_backbone_config = getattr(self, 'point_backbone_config', None)

        if point_backbone is not None and (input_ids.shape[1] != 1 or self.training) and point_clouds is not None:
            # * enter when training or the first generation step of inference
            with torch.no_grad() if self.fix_pointnet else nullcontext():
                if self.fix_pointnet:
                    self.point_backbone.eval()
                if type(point_clouds) is list:
                    # * variable numbers of points
                    point_features = []
                    for point_cloud in point_clouds: # * iterate over batch
                        point_feature = self.point_backbone(point_cloud.unsqueeze(0))[0]
                        point_features.append(point_feature)
                else:
                    point_features = self.point_backbone(point_clouds)

            if type(point_clouds) is list:
                point_features = [self.point_proj(point_feature) for point_feature in point_features]
            else:
                point_features = self.point_proj(point_features)

            dummy_point_features = torch.zeros(point_backbone_config['point_token_len'], point_backbone_config['backbone_output_dim'], device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_point_features = self.point_proj(dummy_point_features)

            new_input_embeds = []
            cur_point_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds): # * input_ids: B, L; input_embeds: B, L, C
                if (cur_input_ids == point_backbone_config['point_patch_token']).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_point_features).sum() # * do nothing
                    new_input_embeds.append(cur_input_embeds)
                    cur_point_idx += 1
                    continue
                cur_point_features = point_features[cur_point_idx].to(device=cur_input_embeds.device)
                num_patches = cur_point_features.shape[0] # * number of point tokens
                if point_backbone_config['mm_use_point_start_end']:
                    if (cur_input_ids == point_backbone_config["point_start_token"]).sum() != (cur_input_ids == point_backbone_config["point_end_token"]).sum():
                        raise ValueError("The number of point start tokens and point end tokens should be the same.")
                    point_start_tokens = torch.where(cur_input_ids == point_backbone_config["point_start_token"])[0]
                    for point_start_token_pos in point_start_tokens:
                        if cur_input_ids[point_start_token_pos + num_patches + 1] != point_backbone_config["point_end_token"]:
                            raise ValueError("The point end token should follow the point start token.")
                        if orig_embeds_params is not None: # * will not update the original embeddings except for POINT_START_TOKEN and POINT_END_TOKEN
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos].detach(), cur_input_embeds[point_start_token_pos:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:point_start_token_pos + num_patches + 2], cur_input_embeds[point_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:]), dim=0)
                        cur_point_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    if (cur_input_ids == point_backbone_config["point_patch_token"]).sum() != num_patches:
                        raise ValueError("The number of point patch tokens should be the same as the number of point patches.")
                    masked_indices = torch.where(cur_input_ids == point_backbone_config["point_patch_token"])[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The point patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_point_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_point_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_point_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(PointLLMLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

VAE_CONFIG = {'dino_version': 'mv-sd-dit-dynaInp-trilatent', 
              'encoder_in_channels': 10, 
              'img_size': [256], 
              'patch_size': 14, 
              'in_chans': 384, 
              'num_classes': 0, 
              'embed_dim': 384, 
              'depth': 6, 
              'num_heads': 16, 
              'mlp_ratio': 4, 
              'qkv_bias': False, 
              'qk_scale': None, 
              'drop_rate': 0.1, 
              'attn_drop_rate': 0.0, 
              'drop_path_rate': 0.0, 
              'norm_layer': 'nn.LayerNorm', 
              'cls_token': False, 
              'encoder_cls_token': False, 
              'decoder_cls_token': False, 
              'sr_kwargs': {}, 
              'sr_ratio': 2, 
              'use_clip': False, 
              'arch_encoder': 'vits', 
              'arch_decoder': 'vitb', 
              'load_pretrain_encoder': False, 
              'encoder_lr': 0.0001, 
              'encoder_weight_decay': 0.001, 
              'no_dim_up_mlp': True, 
              'dim_up_mlp_as_func': False, 
              'decoder_load_pretrained': False, 
              'uvit_skip_encoder': False, 
              'vae_p': 2, 
              'ldm_z_channels': 4, 
              'ldm_embed_dim': 4, 
              'use_conf_map': False, 
              'sd_E_ch': 64, 
              'z_channels': 12, 
              'sd_E_num_res_blocks': 1, 
              'arch_dit_decoder': 'DiT2-B/2', 
              'return_all_dit_layers': False, 
              'lrm_decoder': False, 
              'gs_rendering': False, 
              'decomposed': True, 
              'triplane_fg_bg': False, 
              'cfg': 'objverse_tuneray_aug_resolution_64_64_auto', 
              'density_reg': 0.0, 
              'density_reg_p_dist': 0.004, 
              'reg_type': 'l1', 
              'triplane_decoder_lr': 0.0001, 
              'super_resolution_lr': 0.0001, 
              'c_scale': 1, 
              'nsr_lr': 0.02, 
              'triplane_size': 224, 
              'decoder_in_chans': 32, 
              'triplane_in_chans': -1, 
              'decoder_output_dim': 3, 
              'out_chans': 96, 
              'c_dim': 25, 
              'ray_start': 0.6, 
              'ray_end': 1.8, 
              'rendering_kwargs': {'image_resolution': 256, 
                                   'disparity_space_sampling': False, 
                                   'clamp_mode': 'softplus', 
                                   'c_gen_conditioning_zero': True, 
                                   'c_scale': 1, 
                                   'superresolution_noise_mode': 'none', 
                                   'density_reg': 0.0, 
                                   'density_reg_p_dist': 0.004, 
                                   'reg_type': 'l1', 
                                   'decoder_lr_mul': 1, 
                                   'decoder_activation': 'sigmoid', 
                                   'sr_antialias': True, 
                                   'return_triplane_features': False, 
                                   'return_sampling_details_flag': True, 
                                   'superresolution_module': 'utils.torch_utils.components.NearestConvSR', 
                                   'depth_resolution': 48, 
                                   'depth_resolution_importance': 48, 
                                   'ray_start': 'auto', 
                                   'ray_end': 'auto', 
                                   'box_warp': 0.9, 
                                   'white_back': True, 
                                   'radius_range': [1.5, 2], 
                                   'sampler_bbox_min': -0.45, 
                                   'sampler_bbox_max': 0.45, 
                                   'filter_out_of_bbox': True, 
                                   'PatchRaySampler': True, 
                                   'patch_rendering_resolution': 48, 
                                   'z_near': 1.05, 
                                   'z_far': 2.45, 
                                   'grid_res': 128, 
                                   'grid_scale': 2.005}, 
                'sr_training': False, 
                'bcg_synthesis': False, 
                'bcg_synthesis_kwargs': {}, 
                'image_size': 256, 
                'patch_rendering_resolution': 48, 
                'vit_decoder_lr': 5e-05, 
                'vit_decoder_wd': 0.001, 
                'ae_classname': 'pointllm.model.vit.vit_triplane.ft'}

class AE(torch.nn.Module):

    def __init__(self,
                 encoder,
                 decoder,
                 img_size,
                 encoder_cls_token,
                 decoder_cls_token,
                 preprocess,
                 use_clip,
                 dino_version='v1',
                 clip_dtype=None,
                 no_dim_up_mlp=False,
                 dim_up_mlp_as_func=False,
                 uvit_skip_encoder=False,
                 confnet=None) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.img_size = img_size
        self.encoder_cls_token = encoder_cls_token
        self.decoder_cls_token = decoder_cls_token
        self.use_clip = use_clip
        self.dino_version = dino_version
        self.confnet = confnet

        if self.dino_version == 'v2':
            self.encoder.mask_token = None
            self.decoder.vit_decoder.mask_token = None
        # st()
        # False here
        if 'sd' not in self.dino_version:

            self.uvit_skip_encoder = uvit_skip_encoder
            if uvit_skip_encoder:
                logger.log(
                    f'enables uvit: length of vit_encoder.blocks: {len(self.encoder.blocks)}'
                )
                for blk in self.encoder.blocks[len(self.encoder.blocks) // 2:]:
                    blk.skip_linear = nn.Linear(2 * self.encoder.embed_dim,
                                                self.encoder.embed_dim)

                    # trunc_normal_(blk.skip_linear.weight, std=.02)
                    nn.init.constant_(blk.skip_linear.weight, 0)
                    if isinstance(
                            blk.skip_linear,
                            nn.Linear) and blk.skip_linear.bias is not None:
                        nn.init.constant_(blk.skip_linear.bias, 0)
            else:
                logger.log(f'disable uvit')
        else:
            # False here
            if 'dit' not in self.dino_version:  # dino vit, not dit
                self.decoder.vit_decoder.cls_token = None
                self.decoder.vit_decoder.patch_embed.proj = nn.Identity()
                self.decoder.triplane_decoder.planes = None
                self.decoder.vit_decoder.mask_token = None
            # False here
            if self.use_clip:
                self.clip_dtype = clip_dtype  # torch.float16

            else:
                # False here
                if not no_dim_up_mlp and self.encoder.embed_dim != self.decoder.vit_decoder.embed_dim:
                    self.dim_up_mlp = nn.Linear(
                        self.encoder.embed_dim,
                        self.decoder.vit_decoder.embed_dim)
                    # logger.log(
                    #     f"dim_up_mlp: {self.encoder.embed_dim} -> {self.decoder.vit_decoder.embed_dim}, as_func: {self.dim_up_mlp_as_func}"
                    # )
                else:
                    pass
                    # logger.log('ignore dim_up_mlp: ', no_dim_up_mlp)

        self.preprocess = preprocess

        self.dim_up_mlp = None  # CLIP/B-16
        self.dim_up_mlp_as_func = dim_up_mlp_as_func

        # * remove certain components to make sure no unused parameters during DDP
        # self.decoder.vit_decoder.cls_token = nn.Identity()
        torch.cuda.empty_cache()
        # self.decoder.vit_decoder.patch_embed.proj.bias = nn.Identity()
        # self.decoder.vit_decoder.patch_embed.proj.weight = nn.Identity()
        # self.decoder.vit_decoder.patch_embed.proj.bias = nn.Identity()

    def encode(self, *args, **kwargs):
        # st()
        # true here
        if not self.use_clip:
            # false here
            if self.dino_version == 'v1':
                latent = self.encode_dinov1(*args, **kwargs)
            # false here
            elif self.dino_version == 'v2':
                if self.uvit_skip_encoder:
                    latent = self.encode_dinov2_uvit(*args, **kwargs)
                else:
                    latent = self.encode_dinov2(*args, **kwargs)
            # true here
            else:
                latent = self.encoder(*args)

        else:
            latent = self.encode_clip(*args, **kwargs)
        # st()
        return latent

    def encode_dinov1(self, x):
        # return self.encoder(img)
        x = self.encoder.prepare_tokens(x)
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        if not self.encoder_cls_token:
            return x[:, 1:]

        return x

    def encode_dinov2(self, x):
        # return self.encoder(img)
        x = self.encoder.prepare_tokens_with_masks(x, masks=None)
        for blk in self.encoder.blocks:
            x = blk(x)
        x_norm = self.encoder.norm(x)

        if not self.encoder_cls_token:
            return x_norm[:, 1:]
        # else:
        # return x_norm[:, :1]

        # return {
        #     "x_norm_clstoken": x_norm[:, 0],
        #     "x_norm_patchtokens": x_norm[:, 1:],
        # }

        return x_norm

    def encode_dinov2_uvit(self, x):
        # return self.encoder(img)
        x = self.encoder.prepare_tokens_with_masks(x, masks=None)

        # for blk in self.encoder.blocks:
        #     x = blk(x)

        skips = [x]

        # in blks
        for blk in self.encoder.blocks[0:len(self.encoder.blocks) // 2 - 1]:
            x = blk(x)  # B 3 N C
            skips.append(x)

        # mid blks
        for blk in self.encoder.blocks[len(self.encoder.blocks) // 2 -
                                       1:len(self.encoder.blocks) // 2]:
            x = blk(x)  # B 3 N C

        # out blks
        for blk in self.encoder.blocks[len(self.encoder.blocks) // 2:]:
            x = x + blk.skip_linear(torch.cat(
                [x, skips.pop()], dim=-1))  # long skip connections in uvit
            x = blk(x)  # B 3 N C

        x_norm = self.encoder.norm(x)

        if not self.decoder_cls_token:
            return x_norm[:, 1:]

        return x_norm

    def encode_clip(self, x):
        # * replace with CLIP encoding pipeline
        # return self.encoder(img)
        # x = x.dtype(self.clip_dtype)
        x = self.encoder.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            self.encoder.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.encoder.positional_embedding.to(x.dtype)
        x = self.encoder.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.encoder.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.encoder.ln_post(x[:, 1:, :])  # * return the spatial tokens

        return x

        # x = self.ln_post(x[:, 0, :]) # * return the spatial tokens

        # if self.proj is not None:
        #     x = x @ self.proj

        # return x

    def decode_wo_triplane(self, latent, c=None, img_size=None):
        # st()
        if img_size is None:
            img_size = self.img_size

        # False here
        if self.dim_up_mlp is not None:
            if not self.dim_up_mlp_as_func:
                latent = self.dim_up_mlp(latent)
                # return self.decoder.vit_decode(latent, img_size)
            else:
                return self.decoder.vit_decode(
                    latent, img_size,
                    dim_up_mlp=self.dim_up_mlp)  # used in vae-ldm
        # st()
        from vit.vit_triplane import RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_S
        RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_S.vit_decode
        return self.decoder.vit_decode(latent, img_size, c=c)

    def decode(self, latent, c, img_size=None, return_raw_only=False):
        # if img_size is None:
        #     img_size = self.img_size

        # if self.dim_up_mlp is not None:
        #     latent = self.dim_up_mlp(latent)

        latent = self.decode_wo_triplane(latent, img_size=img_size, c=c)
        # return self.decoder.triplane_decode(latent, c, return_raw_only=return_raw_only)
        return self.decoder.triplane_decode(latent, c)

    def decode_after_vae_no_render(
        self,
        ret_dict,
        img_size=None,
    ):

        if img_size is None:
            img_size = self.img_size

        assert self.dim_up_mlp is None
        # if not self.dim_up_mlp_as_func:
        #     latent = self.dim_up_mlp(latent)
        # return self.decoder.vit_decode(latent, img_size)

        latent = self.decoder.vit_decode_backbone(ret_dict, img_size)
        ret_dict = self.decoder.vit_decode_postprocess(latent, ret_dict)
        return ret_dict

    def decode_after_vae(
            self,
            #  latent,
            ret_dict,  # vae_dict
            c,
            img_size=None,
            return_raw_only=False):
        ret_dict = self.decode_after_vae_no_render(ret_dict, img_size)
        return self.decoder.triplane_decode(ret_dict, c)

    def decode_confmap(self, img):
        assert self.confnet is not None
        # https://github.com/elliottwu/unsup3d/blob/dc961410d61684561f19525c2f7e9ee6f4dacb91/unsup3d/model.py#L152
        # conf_sigma_l1 = self.confnet(img)  # Bx2xHxW
        return self.confnet(img)  # Bx1xHxW

    def encode_decode(self, img, c, return_raw_only=False):
        latent = self.encode(img)
        pred = self.decode(latent, c, return_raw_only=return_raw_only)
        if self.confnet is not None:
            pred.update({
                'conf_sigma': self.decode_confmap(img)  # 224x224
            })

        return pred
    

    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:    # return List[Bl]
        ret_dict = self.decoder.vae_reparameterization(self.encode(inp_img_no_grad), None)
        # False here
        # if isinstance(ret_dict, dict):
        #     latent_to_quantize = ret_dict['latent_normalized_2Ddiffusion']  # B, C*3, H, W
        f = self.decoder.superresolution['quant_conv'](ret_dict['latent_normalized_2Ddiffusion'])
        from pointllm.model.vit.quant import VectorQuantizer2
        VectorQuantizer2.f_to_idxBl_or_fhat
        return self.decoder.superresolution['quantize'].f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)
    def forward(self,
                img=None,
                c=None,
                latent=None,
                behaviour='enc_dec',
                coordinates=None,
                directions=None,
                return_raw_only=False,
                *args,
                **kwargs):
        """wrap all operations inside forward() for DDP use.
        """
        # st()
        # behaviour == 'enc_dec_wo_triplane' here
        if behaviour == 'enc_dec':
            pred = self.encode_decode(img, c, return_raw_only=return_raw_only)
            return pred

        elif behaviour == 'enc':
            latent = self.encode(img)
            return latent

        elif behaviour == 'dec':
            assert latent is not None
            pred: dict = self.decode(latent,
                                     c,
                                     self.img_size,
                                     return_raw_only=return_raw_only)
            return pred

        elif behaviour == 'dec_wo_triplane':
            assert latent is not None
            st()
            pred: dict = self.decode_wo_triplane(latent, self.img_size)
            return pred

        elif behaviour == 'enc_dec_wo_triplane':
            latent = self.encode(img)
            # st()
            pred: dict = self.decode_wo_triplane(latent, img_size=self.img_size, c=c)
            return pred

        elif behaviour == 'encoder_vae':
            latent = self.encode(img)
            ret_dict = self.decoder.vae_reparameterization(latent, True)
            return ret_dict

        elif behaviour == 'decode_after_vae_no_render':
            pred: dict = self.decode_after_vae_no_render(latent, self.img_size)
            return pred

        elif behaviour == 'decode_after_vae':
            pred: dict = self.decode_after_vae(latent, c, self.img_size)
            return pred

        # elif behaviour == 'gaussian_dec':
        #     assert latent is not None
        #     pred: dict = self.decoder.triplane_decode(
        #         latent, c, return_raw_only=return_raw_only, **kwargs)
        #     # pred: dict = self.decoder.triplane_decode(latent, c)

        elif behaviour == 'triplane_dec':
            assert latent is not None
            # st()
            from vit.vit_triplane import RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_S
            RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_S.triplane_decode
            pred: dict = self.decoder.triplane_decode(
                latent, c, return_raw_only=return_raw_only, **kwargs)
            # pred: dict = self.decoder.triplane_decode(latent, c)

        elif behaviour == 'triplane_decode_grid':
            assert latent is not None
            from vit.vit_triplane import RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_S
            RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_S.triplane_decode_grid
            pred: dict = self.decoder.triplane_decode_grid(
                latent, **kwargs)
            # pred: dict = self.decoder.triplane_decode(latent, c)

        elif behaviour == 'vit_postprocess_triplane_dec':
            assert latent is not None
            latent = self.decoder.vit_decode_postprocess(
                latent)  # translate spatial token from vit-decoder into 2D
            pred: dict = self.decoder.triplane_decode(
                latent, c)  # render with triplane

        elif behaviour == 'triplane_renderer':
            assert latent is not None
            pred: dict = self.decoder.triplane_renderer(
                latent, coordinates, directions)

        # elif behaviour == 'triplane_SR':
        #     assert latent is not None
        #     pred: dict = self.decoder.triplane_renderer(
        #         latent, coordinates, directions)

        elif behaviour == 'get_rendering_kwargs':
            pred = self.decoder.triplane_decoder.rendering_kwargs

        return pred
def create_3DAE_model(
        arch_encoder,
        arch_decoder,
        dino_version='v1',
        img_size=[224],
        patch_size=16,
        in_chans=384,
        num_classes=0,
        embed_dim=1024,  # Check ViT encoder dim
        depth=6,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        # norm_layer=nn.LayerNorm,
        norm_layer='nn.LayerNorm',
        out_chans=96,
        decoder_in_chans=32,
        triplane_in_chans=-1,
        decoder_output_dim=32,
        encoder_cls_token=False,
        decoder_cls_token=False,
        c_dim=25,  # Conditioning label (C) dimensionality.
        image_size=128,  # Output resolution.
        img_channels=3,  # Number of output color channels.
        rendering_kwargs={},
        load_pretrain_encoder=False,
        decomposed=True,
        triplane_size=224,
        ae_classname='ViTTriplaneDecomposed',
        use_clip=False,
        sr_kwargs={},
        sr_ratio=2,
        no_dim_up_mlp=False,
        dim_up_mlp_as_func=False,
        decoder_load_pretrained=True,
        uvit_skip_encoder=False,
        bcg_synthesis_kwargs={},
        # decoder params
        vae_p=1,
        ldm_z_channels=4,
        ldm_embed_dim=4,
        use_conf_map=False,
        triplane_fg_bg=False,
        encoder_in_channels=3,
        sd_E_ch=64,
        z_channels=3*4,
        sd_E_num_res_blocks=1,
        arch_dit_decoder='DiT2-B/2',
        lrm_decoder=False,
        gs_rendering=False,
        return_all_dit_layers=False,
        *args,
        **kwargs):

    # TODO, check pre-trained ViT encoder cfgs

    preprocess = None
    clip_dtype = None
    # st()
    # False here
    if load_pretrain_encoder:
        if not use_clip:
            if dino_version == 'v1':
                encoder = torch.hub.load(
                    'facebookresearch/dino:main',
                    'dino_{}{}'.format(arch_encoder, patch_size))
                logger.log(
                    f'loaded pre-trained dino v1 ViT-S{patch_size} encoder ckpt'
                )
            elif dino_version == 'v2':
                encoder = torch.hub.load(
                    'facebookresearch/dinov2',
                    'dinov2_{}{}'.format(arch_encoder, patch_size))
                logger.log(
                    f'loaded pre-trained dino v2 {arch_encoder}{patch_size} encoder ckpt'
                )
            elif 'sd' in dino_version:  # just for compat

                if 'mv' in dino_version:
                    if 'lgm' in dino_version:
                        encoder_cls = MVUNet(
                            input_size=256,
                            up_channels=(1024, 1024, 512, 256,
                                         128),  # one more decoder
                            up_attention=(True, True, True, False, False),
                            splat_size=128,
                            output_size=
                            512,  # render & supervise Gaussians at a higher resolution.
                            batch_size=8,
                            num_views=8,
                            gradient_accumulation_steps=1,
                            # mixed_precision='bf16',
                        )
                    # elif 'gs' in dino_version:
                    #     encoder_cls = MVEncoder
                    # else:
                    #     encoder_cls = MVEncoder
                                # else:
                    if 'dynaInp' in dino_version:
                        # if 'ca' in dino_version:
                        #     encoder_cls = MVEncoderGSDynamicInp_CA
                        # else:
    
                        encoder_cls = MVEncoderGSDynamicInp
                    else:
                        encoder_cls = MVEncoder
                    attn_kwargs = {
                        'n_heads': 8,
                        'd_head': 64,
                    }


                else:
                    encoder_cls = Encoder

                # VAR Version
                # encoder = encoder_cls(  # mono input
                #     double_z=True,
                #     resolution=256,
                #     in_channels=encoder_in_channels,
                #     # ch=128,
                #     ch=64,  # ! fit in the memory
                #     # ch_mult=[1,2,4,4],
                #     # num_res_blocks=2,
                #     ch_mult=[1, 2, 4, 4],
                #     num_res_blocks=1,
                #     dropout=0.0,
                #     attn_resolutions=[],
                #     out_ch=3,  # unused
                #     z_channels=4 * 3,
                # )  # stable diffusion encoder

            else:
                raise NotImplementedError()

        else:
            import clip
            model, preprocess = clip.load("ViT-B/16", device=dist_util.dev())
            model.float()  # convert weight to float32
            clip_dtype = model.dtype
            encoder = getattr(
                model, 'visual')  # only use the CLIP visual encoder here
            encoder.requires_grad_(False)
            logger.log(
                f'loaded pre-trained CLIP ViT-B{patch_size} encoder, fixed.')
    # True here
    elif 'sd' in dino_version:
        attn_kwargs = {}
        # True here
        if 'mv' in dino_version:
            # False here
            if 'lgm' in dino_version:
                encoder = LGM_MVEncoder(
                    in_channels=9,
                    # input_size=256,
                    up_channels=(1024, 1024, 512, 256,
                                 128),  # one more decoder
                    up_attention=(True, True, True, False, False),
                    # splat_size=128,
                    # output_size=
                    # 512,  # render & supervise Gaussians at a higher resolution.
                    # batch_size=8,
                    # num_views=8,
                    # gradient_accumulation_steps=1,
                    # mixed_precision='bf16',
                )
            # False here
            elif 'gs' in dino_version:
                print('using MVEncoderGS')
                encoder_cls = MVEncoderGS
                attn_kwargs = {
                    'n_heads': 8,
                    'd_head': 64,
                }

            else:
                # 0523 version: use this encoder
                print('using MVEncoder')
                encoder_cls = MVEncoder
                attn_kwargs = {
                    'n_heads': 8,
                    'd_head': 64,
                }

        else:
            print('using Encoder')
            encoder_cls = Encoder

        # st()
        # True here
        Cvae = 8
        if 'lgm' not in dino_version: # TODO, for compat now
            # st()
            # TODO (chen): set this argument in sh file
            double_z = False
            # TODO_Done (chen): see the structure of the encoder, and compare with the VQVAE
            # This encoder is similar to the VQVAE encoder, but adds a fusion layer to fuse multiview image features

            # VAR
            encoder = encoder_cls(
                double_z=double_z,
                resolution=256,
                in_channels=encoder_in_channels,
                # ch=128,
                # ch=64, # ! fit in the memory
                ch=160,
                # ch_mult=[1,2,4,4],
                # num_res_blocks=2,
                ch_mult=[1, 1, 2, 2, 4],
                # num_res_blocks=1,
                num_res_blocks=2,
                dropout=0.0,
                attn_resolutions=[],
                out_ch=3,  # unused
                # z_channels=32 * 3, # 4 * 3
                # z_channels=16 *3,
                z_channels= Cvae * 3,
                attn_kwargs=attn_kwargs,
            )  # stable diffusion encoder

            # finetune version
            # encoder = encoder_cls(
            #     double_z=double_z,
            #     resolution=256,
            #     in_channels=encoder_in_channels,
            #     # ch=128,
            #     ch=64, # ! fit in the memory
            #     # ch=160,
            #     ch_mult=[1,2,4,4],
            #     # num_res_blocks=2,
            #     # ch_mult=[1, 1, 2, 2, 4],
            #     num_res_blocks=1,
            #     # num_res_blocks=2,
            #     dropout=0.0,
            #     attn_resolutions=[],
            #     out_ch=3,  # unused
            #     z_channels=32 * 3, # 4 * 3
            #     # z_channels=4 * 3, # 4 * 3
            #     # z_channels=16 *3,
            #     attn_kwargs=attn_kwargs,
            # )  # stable diffusion encoder


    else:
        encoder = vits.__dict__[arch_encoder](
            patch_size=patch_size,
            drop_path_rate=drop_path_rate,  # stochastic depth
            img_size=img_size)

    # assert decomposed
    # if decomposed:
    if triplane_in_chans == -1:
        triplane_in_chans = decoder_in_chans

    # if triplane_fg_bg:
    #     triplane_renderer_cls = Triplane_fg_bg_plane
    # else:
    # TODO (chen) Done: see the structure of Triplane decoder
    # render the 3D scene according to given camera parameters
    # including depth, rgb, weight (mask) 
    triplane_renderer_cls = Triplane

    # triplane_decoder = Triplane(
    triplane_decoder = triplane_renderer_cls(
        c_dim,  # Conditioning label (C) dimensionality.
        image_size,  # Output resolution.
        img_channels,  # Number of output color channels.
        rendering_kwargs=rendering_kwargs,
        out_chans=out_chans,
        # create_triplane=True,  # compatability, remove later
        triplane_size=triplane_size,
        decoder_in_chans=triplane_in_chans,
        decoder_output_dim=decoder_output_dim,
        sr_kwargs=sr_kwargs,
        bcg_synthesis_kwargs=bcg_synthesis_kwargs,
        lrm_decoder=lrm_decoder)
    # st()
    # False here
    if load_pretrain_encoder:

        if dino_version == 'v1':
            vit_decoder = torch.hub.load(
                'facebookresearch/dino:main',
                'dino_{}{}'.format(arch_decoder, patch_size))
            logger.log(
                'loaded pre-trained decoder',
                "facebookresearch/dino:main', 'dino_{}{}".format(
                    arch_decoder, patch_size))
        else:

            vit_decoder = torch.hub.load(
                'facebookresearch/dinov2',
                # 'dinov2_{}{}'.format(arch_decoder, patch_size))
                'dinov2_{}{}'.format(arch_decoder, patch_size),
                pretrained=decoder_load_pretrained)
            logger.log(
                'loaded pre-trained decoder',
                "facebookresearch/dinov2', 'dinov2_{}{}".format(
                    arch_decoder,
                    patch_size), 'pretrianed=', decoder_load_pretrained)
    # True here
    elif 'dit' in dino_version:
        from .dit.dit_decoder import DiT2_models

        # TODO (chen) done: see the structure of the decoder, and compare with the VQVAE
        # Transformer (used to perform self attention and global attention in latent 3D plane) -> upsampler (upsample to final Triplane)
        # arch_dit_decoder == 'DiT2-B/2' here
        # st()
        vit_decoder = DiT2_models[arch_dit_decoder](
            input_size=16,
            num_classes=0,
            learn_sigma=False,
            in_channels=embed_dim,
            mixed_prediction=False,
            context_dim=None,  # add CLIP text embedding
            roll_out=True, plane_n=4 if 
            'gs' in dino_version else 3,
            return_all_layers=return_all_dit_layers,
            )

    else:  # has bug on global token, to fix
        vit_decoder = vits.__dict__[arch_decoder](
            patch_size=patch_size,
            drop_path_rate=drop_path_rate,  # stochastic depth
            img_size=img_size)

    # decoder = ViTTriplaneDecomposed(vit_decoder, triplane_decoder)
    # if True:
    # st()
    decoder_kwargs = dict(
        class_name=ae_classname,
        vit_decoder=vit_decoder,
        triplane_decoder=triplane_decoder,
        # encoder_cls_token=encoder_cls_token,
        cls_token=decoder_cls_token,
        sr_ratio=sr_ratio,
        vae_p=vae_p,
        ldm_z_channels=ldm_z_channels,
        ldm_embed_dim=ldm_embed_dim,
    )
    # TODO (chen) done: see the function, how to construct the class
    # The decoder here is RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_S
    # pls see the structure of this class
    # st()
    from pointllm.model import dnnlib
    decoder = dnnlib.util.construct_class_by_name(**decoder_kwargs)


    # if return_encoder_decoder:
    #     return encoder, decoder, img_size[0], cls_token
    # else:

    # False here
    if use_conf_map:
        # confnet = ConfNet(cin=3, cout=1, nf=64, zdim=128)
        raise NotImplementedError()
    else:
        confnet = None

    # TODO (chen) done: see the structure of the AE
    # AE incorporates the encoder and decoder, and define different manipulation functions in forward function
    auto_encoder = AE(
        encoder,
        decoder,
        img_size[0],
        encoder_cls_token,
        decoder_cls_token,
        preprocess,
        use_clip,
        dino_version,
        clip_dtype,
        no_dim_up_mlp=no_dim_up_mlp,
        dim_up_mlp_as_func=dim_up_mlp_as_func,
        uvit_skip_encoder=uvit_skip_encoder,
        confnet=confnet,
    )

    # logger.log(auto_encoder)
    torch.cuda.empty_cache()

    return auto_encoder
class PointLLMLlamaModelVAR(LlamaModel):
    config_class = PointLLMConfig 

    def __init__(self, config: LlamaConfig):
        super(PointLLMLlamaModelVAR, self).__init__(config)

        self.point_backbone_type = config.point_backbone
        logger.info(f"Using {self.point_backbone_type}.")

        if self.point_backbone_type == "PointBERT":
            # initialize VQVAE model
            # st()
            self.point_backbone = create_3DAE_model(**VAE_CONFIG)
            # st()
            # from pointllm.model import PointTransformer
            # # address of config file, in the same dir of this file
            point_bert_config_name = getattr(config, "point_backbone_config_name", "PointTransformer_8192point_2layer") # * default for v1.2, v1.1 uses PointTransformer_base_8192point.yaml
            point_bert_config_addr = os.path.join(os.path.dirname(__file__), "pointbert", f"{point_bert_config_name}.yaml")
            # print(f"Loading PointBERT config from {point_bert_config_addr}.")
            point_bert_config = cfg_from_yaml_file(point_bert_config_addr)
            # if getattr(config, "use_color", False):
            #     point_bert_config.model.point_dims = 6
            # use_max_pool = getattr(point_bert_config.model, "use_max_pool", False) # * default is false
            
            # self.point_backbone = PointTransformer(point_bert_config.model, use_max_pool=use_max_pool)
            # logger.info(f"Using {self.point_backbone.point_dims} dim of points.")

            # self.point_backbone_config = {
            #     "point_cloud_dim": point_bert_config.model.point_dims,
            #     # "backbone_output_dim": point_bert_config.model.trans_dim if not use_max_pool else point_bert_config.model.trans_dim * 2,
            #     "project_output_dim": self.config.hidden_size,
            #     # "point_token_len": point_bert_config.model.num_group + 1 if not use_max_pool else 1, # * number of output features, with cls token
            #     "mm_use_point_start_end": self.config.mm_use_point_start_end,
            #     "projection_hidden_layer": point_bert_config.model.get('projection_hidden_layer', 0),
            #     # "use_max_pool": use_max_pool
            # }
            from pointllm.data.g_buffer_objaverse import point_backbone_config as predifined_point_backbone_config
            self.point_backbone_config  = predifined_point_backbone_config
            if point_bert_config.model.get('projection_hidden_layer', 0) > 0:
                self.point_backbone_config["projection_hidden_dim"] = point_bert_config.model.projection_hidden_dim # a list
            
            # logger.info(f"Use max pool is {use_max_pool}. Number of point token is {self.point_backbone_config['point_token_len']}.")

        # * print relevant info with projection layers
        # backbone_output_dim = self.point_backbone_config["backbone_output_dim"]
        backbone_output_dim = 8
        logger.info(f"Point backbone output dim: {backbone_output_dim}.")
        logger.info(f"Use {self.point_backbone_config['projection_hidden_layer']} projection hiddent layers.")
        if self.point_backbone_config['projection_hidden_layer'] > 0:
            # Add projection layer with linear layers and GELU activation
            projection_layers = []
            last_dim = backbone_output_dim
            for i in range(point_bert_config.model.projection_hidden_layer):
                projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["projection_hidden_dim"][i]))
                projection_layers.append(nn.GELU())
                last_dim = self.point_backbone_config["projection_hidden_dim"][i]

            projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["project_output_dim"]))
            self.point_proj = nn.Sequential(*projection_layers)
            logger.info(f"Each layer with {point_bert_config.model.projection_hidden_dim} hidden units.")
        else:
            # Single layer
            self.point_proj = nn.Linear(backbone_output_dim, self.point_backbone_config['project_output_dim'])
        logger.info(f"Point projector output dim: {self.point_backbone_config['project_output_dim']}.")

        self.fix_pointnet = False
        self.fix_llm = False
        # st()

    def load_point_backbone_checkpoint(self, checkpoint_path=None):
        # st()
        # checkpoint_path = os.path.join(os.path.dirname(checkpoint_path), "model_rec0350000.pt")
        checkpoint = torch.load(checkpoint_path)
        self.point_backbone.load_state_dict(checkpoint)
        # st()
        # self.point_backbone.load_checkpoint(self.config.point_backbone_ckpt if checkpoint_path is None else checkpoint_path)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # st()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        point_backbone = getattr(self, 'point_backbone', None)
        point_backbone_config = getattr(self, 'point_backbone_config', None)

        if point_backbone is not None and (input_ids.shape[1] != 1 or self.training) and point_clouds is not None:
            # * enter when training or the first generation step of inference
            with torch.no_grad() if self.fix_pointnet else nullcontext():
                if self.fix_pointnet:
                    self.point_backbone.eval()

                # indices = torch.cat([torch.arange(i, i + 6) for i in range(0, point_clouds.size(0), 12)])
                # inp = point_clouds[indices]
                # gt_idx_Bl = self.point_backbone.img_to_idxBl(inp)
                # group_size = 3
                # gt_Bl_concate_list = []
                # for i in range(len(gt_idx_Bl)):
                #     N = gt_idx_Bl[i].shape[-1]
                #     gt_Bl_concate_list.append(gt_idx_Bl[i].reshape(-1, group_size * N))
                # gt_BL = torch.cat(gt_Bl_concate_list, dim=1)
                # gt_BL = gt_BL[:, :point_backbone_config['point_token_len']]
                # st()
                
                gt_BL = point_clouds[:, :point_backbone_config['point_token_len']]
                # st()
                import torch.nn.functional as F
                embeddings = F.normalize(self.point_backbone.decoder.superresolution['quantize'].embedding.weight, p=2, dim=-1)
                point_features = embeddings[gt_BL]
                
                # if type(point_clouds) is list:
                #     # * variable numbers of points
                #     point_features = []
                #     for point_cloud in point_clouds: # * iterate over batch
                #         point_feature = self.point_backbone(point_cloud.unsqueeze(0))[0]
                #         point_features.append(point_feature)
                # else:
                #     point_features = self.point_backbone(point_clouds)

            if type(point_clouds) is list:
                point_features = [self.point_proj(point_feature) for point_feature in point_features]
            else:
                point_features = self.point_proj(point_features)

            dummy_point_features = torch.zeros(point_backbone_config['point_token_len'], point_backbone_config['backbone_output_dim'], device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_point_features = self.point_proj(dummy_point_features)

            new_input_embeds = []
            cur_point_idx = 0
            # st()
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds): # * input_ids: B, L; input_embeds: B, L, C
                if (cur_input_ids == point_backbone_config['point_patch_token']).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_point_features).sum() # * do nothing
                    new_input_embeds.append(cur_input_embeds)
                    cur_point_idx += 1
                    continue
                cur_point_features = point_features[cur_point_idx].to(device=cur_input_embeds.device)
                num_patches = cur_point_features.shape[0] # * number of point tokens
                if point_backbone_config['mm_use_point_start_end']:
                    if (cur_input_ids == point_backbone_config["point_start_token"]).sum() != (cur_input_ids == point_backbone_config["point_end_token"]).sum():
                        raise ValueError("The number of point start tokens and point end tokens should be the same.")
                    point_start_tokens = torch.where(cur_input_ids == point_backbone_config["point_start_token"])[0]
                    for point_start_token_pos in point_start_tokens:
                        if cur_input_ids[point_start_token_pos + num_patches + 1] != point_backbone_config["point_end_token"]:
                            raise ValueError("The point end token should follow the point start token.")
                        if orig_embeds_params is not None: # * will not update the original embeddings except for POINT_START_TOKEN and POINT_END_TOKEN
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos].detach(), cur_input_embeds[point_start_token_pos:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:point_start_token_pos + num_patches + 2], cur_input_embeds[point_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:]), dim=0)
                        cur_point_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    if (cur_input_ids == point_backbone_config["point_patch_token"]).sum() != num_patches:
                        raise ValueError("The number of point patch tokens should be the same as the number of point patches.")
                    masked_indices = torch.where(cur_input_ids == point_backbone_config["point_patch_token"])[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The point patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_point_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_point_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_point_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        # return super(PointLLMLlamaModel, self).forward(
        return super(PointLLMLlamaModelVAR, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
class PointLLMLlamaForCausalLM(LlamaForCausalLM):
    config_class = PointLLMConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        # self.model = PointLLMLlamaModel(config)
        self.model = PointLLMLlamaModelVAR(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None, # * control whether to return past_key_values
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            point_clouds=point_clouds
        )
        # st()
        # print("lllllllllllllllllllllllllllllllllllll")
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous() # * B, L, V(32003)
            shift_labels = labels[..., 1:].contiguous() # * B, L
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "point_clouds": kwargs.get("point_clouds", None),
            }
        )
        return model_inputs

    def initialize_tokenizer_point_backbone_config_wo_embedding(self, tokenizer):
        # * called when stage2 or inference or inference without pre-training, assume tokenizer has point tokens
        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_point_start_end = point_backbone_config['mm_use_point_start_end'] = config.mm_use_point_start_end

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN

        tokenizer.add_tokens([default_point_patch_token], special_tokens=True)

        # * assert tokenizer has the default_point_patch_token
        point_backbone_config['default_point_patch_token'] = default_point_patch_token
        point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids([default_point_patch_token])[0]

        if mm_use_point_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            tokenizer.add_tokens([default_point_start_token, default_point_end_token], special_tokens=True)

            point_backbone_config['default_point_start_token'] = default_point_start_token
            point_backbone_config['default_point_end_token'] = default_point_end_token

            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids([default_point_start_token])[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids([default_point_end_token])[0]
    
    def initialize_tokenizer_point_backbone_config(self, tokenizer, device, fix_llm=True):

        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_point_start_end = point_backbone_config['mm_use_point_start_end'] = config.mm_use_point_start_end

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN
        point_backbone_config['default_point_patch_token'] = default_point_patch_token
        tokenizer.add_tokens([default_point_patch_token], special_tokens=True) # * no need to update embed since it will be replaced
        self.resize_token_embeddings(len(tokenizer)) # ! resize_token_embeddings will make the tokens trainable again
        point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids([default_point_patch_token])[0]

        if mm_use_point_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            point_backbone_config['default_point_start_token'] = default_point_start_token
            point_backbone_config['default_point_end_token'] = default_point_end_token

            num_new_tokens = tokenizer.add_tokens([default_point_start_token, default_point_end_token], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids([default_point_start_token])[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids([default_point_end_token])[0]

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

                # need to update the input embeding, but no need to update the output embedding
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                if fix_llm:
                    self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)] # * only tuning the new embeddings
                    for p in self.get_output_embeddings().parameters(): # * the llm head
                        p.requires_grad = False
                    print(f"Setting output embeddings fixed and {num_new_tokens} new tokens' input embeddings trainable.")
                else:
                    self.get_model().orig_embeds_params = None
                    for p in self.get_output_embeddings().parameters():
                        p.requires_grad = True
                    print("Setting output embeddings and all input embeddings trainable.")

AutoConfig.register("pointllm", PointLLMConfig)
AutoModelForCausalLM.register(PointLLMConfig, PointLLMLlamaForCausalLM)
