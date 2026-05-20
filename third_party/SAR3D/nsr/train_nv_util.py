import copy
import functools
import json
import os
from pathlib import Path
from einops import rearrange
import webdataset as wds

import traceback
import blobfile as bf
import imageio
import numpy as np
import torch as th
import torch.distributed as dist
import torchvision
from PIL import Image
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.nn import update_ema
from guided_diffusion.resample import LossAwareSampler, UniformSampler
from guided_diffusion.train_util import (calc_average_loss,
                                         find_ema_checkpoint,
                                         find_resume_checkpoint,
                                         get_blob_logdir, log_rec3d_loss_dict,
                                         parse_resume_step_from_filename)

from .camera_utils import LookAtPoseSampler, FOV_to_intrinsics

from .train_util import TrainLoop3DRec

import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity


class TrainLoop3DRecNV(TrainLoop3DRec):
    """Supervise the training of novel view."""
    def __init__(self,
                 *,
                 rec_model,
                 loss_class,
                 data,
                 eval_data,
                 batch_size,
                 microbatch,
                 lr,
                 ema_rate,
                 log_interval,
                 eval_interval,
                 save_interval,
                 resume_checkpoint,
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
                         loss_class=loss_class,
                         data=data,
                         eval_data=eval_data,
                         batch_size=batch_size,
                         microbatch=microbatch,
                         lr=lr,
                         ema_rate=ema_rate,
                         log_interval=log_interval,
                         eval_interval=eval_interval,
                         save_interval=save_interval,
                         resume_checkpoint=resume_checkpoint,
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         **kwargs)
        self.rec_cano = True

    def forward_backward(self, batch, *args, **kwargs):
        """Perform forward and backward pass."""
        self.mp_trainer_rec.zero_grad()
        batch_size = batch['img_to_encoder'].shape[0]

        for i in range(0, batch_size, self.microbatch):
            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev())
                for k, v in batch.items()
            }

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):

                target_nvs = {}
                target_cano = {}

                latent = self.rec_model(img=micro['img_to_encoder'],
                                        behaviour='enc_dec_wo_triplane')

                pred = self.rec_model(
                    latent=latent,
                    c=micro['nv_c'],  # predict novel view here
                    behaviour='triplane_dec')

                for k, v in micro.items():
                    if k[:2] == 'nv':
                        orig_key = k.replace('nv_', '')
                        target_nvs[orig_key] = v
                        target_cano[orig_key] = micro[orig_key]

                with self.rec_model.no_sync():  # type: ignore
                    loss, loss_dict, fg_mask = self.loss_class(
                        pred,
                        target_nvs,
                        step=self.step + self.resume_step,
                        test_mode=False,
                        return_fg_mask=True,
                        conf_sigma_l1=None,
                        conf_sigma_percl=None)
                    log_rec3d_loss_dict(loss_dict)

                if self.rec_cano:

                    pred_cano = self.rec_model(latent=latent,
                                               c=micro['c'],
                                               behaviour='triplane_dec')

                    with self.rec_model.no_sync():  # type: ignore

                        fg_mask = target_cano['depth_mask'].unsqueeze(
                            1).repeat_interleave(3, 1).float()

                        loss_cano, loss_cano_dict = self.loss_class.calc_2d_rec_loss(
                            pred_cano['image_raw'],
                            target_cano['img'],
                            fg_mask,
                            step=self.step + self.resume_step,
                            test_mode=False,
                        )

                    loss = loss + loss_cano

                    log_rec3d_loss_dict({
                        f'cano_{k}': v
                        for k, v in loss_cano_dict.items()
                    })

            self.mp_trainer_rec.backward(loss)

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                if self.rec_cano:
                    self.log_img(micro, pred, pred_cano)
                else:
                    self.log_img(micro, pred, None)

    @th.inference_mode()
    def log_img(self, micro, pred, pred_cano):
        """Log images for visualization."""
        def norm_depth(pred_depth):  # to [-1,1]
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())
            return -(pred_depth * 2 - 1)

        pred_img = pred['image_raw']
        gt_img = micro['img']

        gt_depth = micro['depth']
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        gt_depth = norm_depth(gt_depth)

        fg_mask = pred['image_mask'] * 2 - 1  # 0-1
        input_fg_mask = pred_cano['image_mask'] * 2 - 1  # 0-1
        if 'image_depth' in pred:
            pred_depth = norm_depth(pred['image_depth'])
            pred_nv_depth = norm_depth(pred_cano['image_depth'])
        else:
            pred_depth = th.zeros_like(gt_depth)
            pred_nv_depth = th.zeros_like(gt_depth)

        if 'image_sr' in pred:
            if pred['image_sr'].shape[-1] == 512:
                pred_img = th.cat([self.pool_512(pred_img), pred['image_sr']],
                                  dim=-1)
                gt_img = th.cat([self.pool_512(micro['img']), micro['img_sr']],
                                dim=-1)
                pred_depth = self.pool_512(pred_depth)
                gt_depth = self.pool_512(gt_depth)

            elif pred['image_sr'].shape[-1] == 256:
                pred_img = th.cat([self.pool_256(pred_img), pred['image_sr']],
                                  dim=-1)
                gt_img = th.cat([self.pool_256(micro['img']), micro['img_sr']],
                                dim=-1)
                pred_depth = self.pool_256(pred_depth)
                gt_depth = self.pool_256(gt_depth)

            else:
                pred_img = th.cat([self.pool_128(pred_img), pred['image_sr']],
                                  dim=-1)
                gt_img = th.cat([self.pool_128(micro['img']), micro['img_sr']],
                                dim=-1)
                gt_depth = self.pool_128(gt_depth)
                pred_depth = self.pool_128(pred_depth)
        else:
            gt_img = self.pool_64(gt_img)
            gt_depth = self.pool_64(gt_depth)

        pred_vis = th.cat([
            pred_img,
            pred_depth.repeat_interleave(3, dim=1),
            fg_mask.repeat_interleave(3, dim=1),
        ],
                          dim=-1)  # B, 3, H, W

        pred_vis_nv = th.cat([
            pred_cano['image_raw'],
            pred_nv_depth.repeat_interleave(3, dim=1),
            input_fg_mask.repeat_interleave(3, dim=1),
        ],
                             dim=-1)  # B, 3, H, W

        pred_vis = th.cat([pred_vis, pred_vis_nv], dim=-2)  # cat in H dim

        gt_vis = th.cat([
            gt_img,
            gt_depth.repeat_interleave(3, dim=1),
            th.zeros_like(gt_img)
        ],
                        dim=-1)

        if 'conf_sigma' in pred:
            gt_vis = th.cat([gt_vis, fg_mask], dim=-1)  # placeholder

        vis = th.cat([gt_vis, pred_vis], dim=-2)
        vis_tensor = torchvision.utils.make_grid(vis, nrow=vis.shape[-1] //
                                                 64)  # HWC
        torchvision.utils.save_image(
            vis_tensor,
            f'{logger.get_dir()}/{self.step+self.resume_step}.jpg',
            value_range=(-1, 1),
            normalize=True)

        logger.log('log vis to: ',
                   f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')


class TrainLoop3DRecNVPatch(TrainLoop3DRecNV):
    """Add patch rendering."""
    def __init__(self,
                 *,
                 rec_model,
                 loss_class,
                 data,
                 eval_data,
                 batch_size,
                 microbatch,
                 lr,
                 ema_rate,
                 log_interval,
                 eval_interval,
                 save_interval,
                 resume_checkpoint,
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
                         loss_class=loss_class,
                         data=data,
                         eval_data=eval_data,
                         batch_size=batch_size,
                         microbatch=microbatch,
                         lr=lr,
                         ema_rate=ema_rate,
                         log_interval=log_interval,
                         eval_interval=eval_interval,
                         save_interval=save_interval,
                         resume_checkpoint=resume_checkpoint,
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         **kwargs)
        self.eg3d_model = self.rec_model.module.decoder.triplane_decoder  # type: ignore
        self.rec_cano = True

    def forward_backward(self, batch, *args, **kwargs):
        """Perform forward and backward pass with patch sampling."""
        self.mp_trainer_rec.zero_grad()
        batch_size = batch['img_to_encoder'].shape[0]

        for i in range(0, batch_size, self.microbatch):
            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev())
                for k, v in batch.items()
            }

            target = {
                **self.eg3d_model(
                    c=micro['nv_c'],  # type: ignore
                    ws=None,
                    planes=None,
                    sample_ray_only=True,
                    fg_bbox=micro['nv_bbox']),  # rays o / dir
            }

            patch_rendering_resolution = self.eg3d_model.rendering_kwargs[
                'patch_rendering_resolution']  # type: ignore
            cropped_target = {
                k:
                th.empty_like(v)
                [..., :patch_rendering_resolution, :patch_rendering_resolution]
                if k not in [
                    'ins_idx', 'img_to_encoder', 'img_sr', 'nv_img_to_encoder',
                    'nv_img_sr', 'c'
                ] else v
                for k, v in micro.items()
            }

            for j in range(micro['img'].shape[0]):
                top, left, height, width = target['ray_bboxes'][
                    j]  # list of tuple
                for key in ('img', 'depth_mask', 'depth'):  # type: ignore
                    cropped_target[f'{key}'][  # ! no nv_ here
                        j:j + 1] = torchvision.transforms.functional.crop(
                            micro[f'nv_{key}'][j:j + 1], top, left, height,
                            width)

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):

                latent = self.rec_model(img=micro['img_to_encoder'],
                                        behaviour='enc_dec_wo_triplane')

                pred_nv = self.rec_model(
                    latent=latent,
                    c=micro['nv_c'],  # predict novel view here
                    behaviour='triplane_dec',
                    ray_origins=target['ray_origins'],
                    ray_directions=target['ray_directions'],
                )

                with self.rec_model.no_sync():  # type: ignore
                    loss, loss_dict, _ = self.loss_class(pred_nv,
                                                         cropped_target,
                                                         step=self.step +
                                                         self.resume_step,
                                                         test_mode=False,
                                                         return_fg_mask=True,
                                                         conf_sigma_l1=None,
                                                         conf_sigma_percl=None)
                    log_rec3d_loss_dict(loss_dict)

                if self.rec_cano:

                    cano_target = {
                        **self.eg3d_model(
                            c=micro['c'],  # type: ignore
                            ws=None,
                            planes=None,
                            sample_ray_only=True,
                            fg_bbox=micro['bbox']),  # rays o / dir
                    }

                    cano_cropped_target = {
                        k: th.empty_like(v)
                        for k, v in cropped_target.items()
                    }

                    for j in range(micro['img'].shape[0]):
                        top, left, height, width = cano_target['ray_bboxes'][
                            j]  # list of tuple
                        for key in ('img', 'depth_mask',
                                    'depth'):  # type: ignore
                            cano_cropped_target[key][
                                j:j +
                                1] = torchvision.transforms.functional.crop(
                                    micro[key][j:j + 1], top, left, height,
                                    width)

                    pred_cano = self.rec_model(
                        latent=latent,
                        c=micro['c'],
                        behaviour='triplane_dec',
                        ray_origins=cano_target['ray_origins'],
                        ray_directions=cano_target['ray_directions'],
                    )

                    with self.rec_model.no_sync():  # type: ignore

                        fg_mask = cano_cropped_target['depth_mask'].unsqueeze(
                            1).repeat_interleave(3, 1).float()

                        loss_cano, loss_cano_dict = self.loss_class.calc_2d_rec_loss(
                            pred_cano['image_raw'],
                            cano_cropped_target['img'],
                            fg_mask,
                            step=self.step + self.resume_step,
                            test_mode=False,
                        )

                    loss = loss + loss_cano

                    log_rec3d_loss_dict({
                        f'cano_{k}': v
                        for k, v in loss_cano_dict.items()
                    })

            self.mp_trainer_rec.backward(loss)

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                self.log_patch_img(cropped_target, pred_nv, pred_cano)

    @th.inference_mode()
    def log_patch_img(self, micro, pred, pred_cano):
        """Log patch images for visualization."""
        def norm_depth(pred_depth):  # to [-1,1]
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())
            return -(pred_depth * 2 - 1)

        pred_img = pred['image_raw']
        gt_img = micro['img']

        gt_depth = micro['depth']
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        gt_depth = norm_depth(gt_depth)

        fg_mask = pred['image_mask'] * 2 - 1  # 0-1
        input_fg_mask = pred_cano['image_mask'] * 2 - 1  # 0-1
        if 'image_depth' in pred:
            pred_depth = norm_depth(pred['image_depth'])
            pred_cano_depth = norm_depth(pred_cano['image_depth'])
        else:
            pred_depth = th.zeros_like(gt_depth)
            pred_cano_depth = th.zeros_like(gt_depth)

        pred_vis = th.cat([
            pred_img,
            pred_depth.repeat_interleave(3, dim=1),
            fg_mask.repeat_interleave(3, dim=1),
        ],
                          dim=-1)  # B, 3, H, W

        pred_vis_nv = th.cat([
            pred_cano['image_raw'],
            pred_cano_depth.repeat_interleave(3, dim=1),
            input_fg_mask.repeat_interleave(3, dim=1),
        ],
                             dim=-1)  # B, 3, H, W

        pred_vis = th.cat([pred_vis, pred_vis_nv], dim=-2)  # cat in H dim

        gt_vis = th.cat([
            gt_img,
            gt_depth.repeat_interleave(3, dim=1),
            th.zeros_like(gt_img)
        ],
                        dim=-1)

        vis = th.cat([gt_vis, pred_vis], dim=-2)
        vis_tensor = torchvision.utils.make_grid(vis, nrow=vis.shape[-1] //
                                                 64)  # HWC
        torchvision.utils.save_image(
            vis_tensor,
            f'{logger.get_dir()}/{self.step+self.resume_step}.jpg',
            value_range=(-1, 1),
            normalize=True)

        logger.log('log vis to: ',
                   f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')

class TrainLoop3DRecNVPatchSingleForward(TrainLoop3DRecNVPatch):

    def __init__(self,
                 *,
                 rec_model,
                 loss_class,
                 data,
                 eval_data,
                 batch_size,
                 microbatch,
                 lr,
                 ema_rate,
                 log_interval,
                 eval_interval,
                 save_interval,
                 resume_checkpoint,
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
                         loss_class=loss_class,
                         data=data,
                         eval_data=eval_data,
                         batch_size=batch_size,
                         microbatch=microbatch,
                         lr=lr,
                         ema_rate=ema_rate,
                         log_interval=log_interval,
                         eval_interval=eval_interval,
                         save_interval=save_interval,
                         resume_checkpoint=resume_checkpoint,
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         **kwargs)

    def forward_backward(self, batch, *args, **kwargs):
        # Add patch sampling

        self.mp_trainer_rec.zero_grad()
        batch_size = batch['img_to_encoder'].shape[0]

        batch.pop('caption')  # Not required
        batch.pop('ins')  # Not required

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k:
                v[i:i + self.microbatch].to(dist_util.dev()) if isinstance(
                    v, th.Tensor) else v[i:i + self.microbatch]
                for k, v in batch.items()
            }

            # Sample rendering patch
            target = {
                **self.eg3d_model(
                    c=micro['nv_c'],  # type: ignore
                    ws=None,
                    planes=None,
                    sample_ray_only=True,
                    fg_bbox=micro['nv_bbox']),  # Rays o / dir
            }

            patch_rendering_resolution = self.eg3d_model.rendering_kwargs[
                'patch_rendering_resolution']  # type: ignore
            cropped_target = {
                k:
                th.empty_like(v)
                [..., :patch_rendering_resolution, :patch_rendering_resolution]
                if k not in [
                    'ins_idx', 'img_to_encoder', 'img_sr', 'nv_img_to_encoder',
                    'nv_img_sr', 'c', 'caption', 'nv_caption'
                ] else v
                for k, v in micro.items()
            }

            # Crop according to UV sampling
            for j in range(micro['img'].shape[0]):
                top, left, height, width = target['ray_bboxes'][
                    j]  # List of tuple
                for key in ('img', 'depth_mask', 'depth'):  # type: ignore
                    cropped_target[f'{key}'][  # No nv_ here
                        j:j + 1] = torchvision.transforms.functional.crop(
                            micro[f'nv_{key}'][j:j + 1], top, left, height,
                            width)

            # Cano view loss
            cano_target = {
                **self.eg3d_model(
                    c=micro['c'],  # type: ignore
                    ws=None,
                    planes=None,
                    sample_ray_only=True,
                    fg_bbox=micro['bbox']),  # Rays o / dir
            }

            # VIT no AMP
            latent = self.rec_model(img=micro['img_to_encoder'],
                                    behaviour='enc_dec_wo_triplane')

            # Wrap forward within AMP
            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):

                instance_mv_num = batch_size // 4  # 4 pairs by default
                # Roll views for multi-view supervision
                c = th.cat([
                    micro['nv_c'].roll(instance_mv_num * i, dims=0)
                    for i in range(1, 4)
                ])  # Predict novel view here

                ray_origins = th.cat(
                    [
                        target['ray_origins'].roll(instance_mv_num * i, dims=0)
                        for i in range(1, 4)
                    ],
                    0)

                ray_directions = th.cat([
                    target['ray_directions'].roll(instance_mv_num * i, dims=0)
                    for i in range(1, 4)
                ])

                pred_nv_cano = self.rec_model(
                    latent={
                        'latent_after_vit':  # Triplane for rendering
                        latent['latent_after_vit'].repeat(3, 1, 1, 1)
                    },
                    c=c,
                    behaviour='triplane_dec',
                    ray_origins=ray_origins,
                    ray_directions=ray_directions,
                )

                pred_nv_cano.update(latent)
                gt = {
                    k:
                    th.cat(
                        [
                            v.roll(instance_mv_num * i, dims=0)
                            for i in range(1, 4)
                        ],
                        0)
                    for k, v in cropped_target.items()
                }

                with self.rec_model.no_sync():  # type: ignore
                    loss, loss_dict, _ = self.loss_class(
                        pred_nv_cano,
                        gt,  # Prepare merged data
                        step=self.step + self.resume_step,
                        test_mode=False,
                        return_fg_mask=True,
                        conf_sigma_l1=None,
                        conf_sigma_percl=None)
                    log_rec3d_loss_dict(loss_dict)

            self.mp_trainer_rec.backward(loss)

            for name, p in self.rec_model.named_parameters():
                if p.grad is None:
                    logger.log(f"found rec unused param: {name}")
            st()

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                micro_bs = micro['img_to_encoder'].shape[0]
                self.log_patch_img(  # Record one cano view and one novel view
                    cropped_target,
                    {
                        k: pred_nv_cano[k][-micro_bs:]
                        for k in ['image_raw', 'image_depth', 'image_mask']
                    },
                    {
                        k: pred_nv_cano[k][:micro_bs]
                        for k in ['image_raw', 'image_depth', 'image_mask']
                    },
                )

    def eval_loop(self):
        return super().eval_loop()

    @th.inference_mode()
    def eval_novelview_loop_old(self, camera=None):
        # Novel view synthesis given evaluation camera trajectory

        all_loss_dict = []
        novel_view_micro = {}

        export_mesh = True
        if export_mesh:
            Path(f'{logger.get_dir()}/FID_Cals/').mkdir(parents=True,
                                                        exist_ok=True)
        batch = {}


        for eval_idx, render_reference in enumerate(tqdm(self.eval_data)):

            if eval_idx > 500:
                break

            video_out = imageio.get_writer(
                f'{logger.get_dir()}/video_novelview_{self.step+self.resume_step}_{eval_idx}.mp4',
                mode='I',
                fps=25,
                codec='libx264')

            with open(
                    f'{logger.get_dir()}/triplane_{self.step+self.resume_step}_{eval_idx}_caption.txt',
                    'w') as f:
                f.write(render_reference['caption'])

            for key in ['ins', 'bbox', 'caption']:
                if key in render_reference:
                    render_reference.pop(key)

            real_flag = False
            mv_flag = False  # Use full-instance for evaluation? Calculate the metrics.
            if render_reference['c'].shape[:2] == (1, 40):
                real_flag = True
                # Real img monocular reconstruction
                render_reference = [{
                    k: v[0][idx:idx + 1]
                    for k, v in render_reference.items()
                } for idx in range(40)]

            elif render_reference['c'].shape[0] == 8:
                mv_flag = True

                render_reference = {
                    k: v[:4]
                    for k, v in render_reference.items()
                }

                # Save GT
                torchvision.utils.save_image(
                    render_reference[0:4]['img'],
                    logger.get_dir() + '/FID_Cals/{}_inp.png'.format(eval_idx),
                    padding=0,
                    normalize=True,
                    value_range=(-1, 1),
                )

            else:

                render_reference = [{
                    k: v[idx:idx + 1]
                    for k, v in render_reference.items()
                } for idx in range(40)]

                # Single-view version
                render_reference[0]['img_to_encoder'] = render_reference[14][
                    'img_to_encoder']  # Encode side view
                render_reference[0]['img'] = render_reference[14][
                    'img']  # Encode side view

                # Save GT
                torchvision.utils.save_image(
                    render_reference[0]['img'],
                    logger.get_dir() + '/FID_Cals/{}_gt.png'.format(eval_idx),
                    padding=0,
                    normalize=True,
                    value_range=(-1, 1))

            for i, batch in enumerate(render_reference):
                micro = {k: v.to(dist_util.dev()) for k, v in batch.items()}

                if i == 0:
                    if mv_flag:
                        novel_view_micro = None
                    else:
                        novel_view_micro = {
                            k:
                            v[0:1].to(dist_util.dev()).repeat_interleave(
                                micro['img'].shape[0],
                                0) if isinstance(v, th.Tensor) else v[0:1]
                            for k, v in batch.items()
                        }

                else:
                    if i == 1:

                        if export_mesh:

                            mesh_size = 384
                            mesh_thres = 10  # Requires tuning
                            import mcubes
                            import trimesh
                            dump_path = f'{logger.get_dir()}/mesh/'

                            os.makedirs(dump_path, exist_ok=True)

                            grid_out = self.rec_model(
                                latent=pred,
                                grid_size=mesh_size,
                                behaviour='triplane_decode_grid',
                            )

                            vtx, faces = mcubes.marching_cubes(
                                grid_out['sigma'].squeeze(0).squeeze(
                                    -1).cpu().numpy(), mesh_thres)
                            vtx = vtx / (mesh_size - 1) * 2 - 1

                            mesh = trimesh.Trimesh(
                                vertices=vtx,
                                faces=faces,
                            )

                            mesh_dump_path = os.path.join(
                                dump_path, f'{eval_idx}.ply')
                            mesh.export(mesh_dump_path, 'ply')

                            print(f"Mesh dumped to {dump_path}")
                            del grid_out, mesh
                            th.cuda.empty_cache()

                    novel_view_micro = {
                        k:
                        v[0:1].to(dist_util.dev()).repeat_interleave(
                            micro['img'].shape[0], 0)
                        for k, v in novel_view_micro.items()
                    }

                pred = self.rec_model(img=novel_view_micro['img_to_encoder'],
                                      c=micro['c'])  # pred: (B, 3, 64, 64)

                if not real_flag:
                    _, loss_dict = self.loss_class(pred, micro, test_mode=True)
                    all_loss_dict.append(loss_dict)

                pred_depth = pred['image_depth']
                pred_depth = (pred_depth - pred_depth.min()) / (
                    pred_depth.max() - pred_depth.min())
                if 'image_sr' in pred:

                    if pred['image_sr'].shape[-1] == 512:

                        pred_vis = th.cat([
                            micro['img_sr'],
                            self.pool_512(pred['image_raw']), pred['image_sr'],
                            self.pool_512(pred_depth).repeat_interleave(3,
                                                                        dim=1)
                        ],
                                          dim=-1)

                    elif pred['image_sr'].shape[-1] == 256:

                        pred_vis = th.cat([
                            micro['img_sr'],
                            self.pool_256(pred['image_raw']), pred['image_sr'],
                            self.pool_256(pred_depth).repeat_interleave(3,
                                                                        dim=1)
                        ],
                                          dim=-1)

                    else:
                        pred_vis = th.cat([
                            micro['img_sr'],
                            self.pool_128(pred['image_raw']),
                            self.pool_128(pred['image_sr']),
                            self.pool_128(pred_depth).repeat_interleave(3,
                                                                        dim=1)
                        ],
                                          dim=-1)

                else:

                    pooled_depth = self.pool_128(pred_depth).repeat_interleave(
                        3, dim=1)
                    pred_vis = th.cat(
                        [
                            self.pool_128(novel_view_micro['img']),
                            self.pool_128(pred['image_raw']),
                            pooled_depth,
                        ],
                        dim=-1)  # B, 3, H, W

                vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
                vis = vis * 127.5 + 127.5
                vis = vis.clip(0, 255).astype(np.uint8)

                if export_mesh:
                    torchvision.utils.save_image(
                        pred['image_raw'],
                        logger.get_dir() +
                        '/FID_Cals/{}_{}.png'.format(eval_idx, i),
                        padding=0,
                        normalize=True,
                        value_range=(-1, 1))

                    torchvision.utils.save_image(
                        pooled_depth,
                        logger.get_dir() +
                        '/FID_Cals/{}_{}_dpeth.png'.format(eval_idx, i),
                        padding=0,
                        normalize=True,
                        value_range=(0, 1))

                for j in range(vis.shape[0]):
                    video_out.append_data(vis[j])

            video_out.close()

        if not real_flag or mv_flag:
            val_scores_for_logging = calc_average_loss(all_loss_dict)
            with open(os.path.join(logger.get_dir(), 'scores_novelview.json'),
                      'a') as f:
                json.dump({'step': self.step, **val_scores_for_logging}, f)

            for k, v in val_scores_for_logging.items():
                self.writer.add_scalar(f'Eval/NovelView/{k}', v,
                                       self.step + self.resume_step)

        del video_out

        th.cuda.empty_cache()

    @th.inference_mode()
    def eval_novelview_loop(self, camera=None, save_latent=False):
        # Novel view synthesis given evaluation camera trajectory

        if save_latent:  # For diffusion learning
            latent_dir = Path(f'{logger.get_dir()}/latent_dir')
            latent_dir.mkdir(exist_ok=True, parents=True)

        eval_batch_size = 40  # For i23d

        for eval_idx, micro in enumerate(tqdm(self.eval_data)):

            from nsr.script_util import AE
            AE.forward
            latent, vq_loss, usages = self.rec_model(
                img=micro['img_to_encoder'][0:6],
                behaviour='enc_dec_wo_triplane')  # pred: (B, 3, 64, 64)


            if micro['img'].shape[0] == 40:
                assert eval_batch_size == 40

            if save_latent:

                latent_save_dir = f'{logger.get_dir()}/latent_dir/{micro["ins"][0]}'
                Path(latent_save_dir).mkdir(parents=True, exist_ok=True)

                np.save(f'{latent_save_dir}/latent.npy',
                        latent[self.latent_name][0].cpu().numpy())
                assert all([
                    micro['ins'][0] == micro['ins'][i]
                    for i in range(micro['c'].shape[0])
                ])  # Assert same instance

            print(micro['c'][0:6].shape)
            if eval_idx < 100:
                self.render_img_given_triplane(
                    latent['latent_after_vit'],  # B 96 128 128
                    self.rec_model,  # Compatible with join_model
                    name_prefix=f'{eval_idx}',
                    save_img=False,
                    render_reference={'c': micro['c'][0:6]},
                    save_mesh=True,
                    img=micro)

            if eval_idx < 100:
                self.render_video_given_triplane(
                    latent['latent_after_vit'],  # B 96 256 256
                    self.rec_model,  # Compatible with join_model
                    name_prefix=f'{self.step + self.resume_step}_{eval_idx}',
                    save_img=False,
                    render_reference={'c': camera},
                    save_mesh=True)
            else:
                break
        
        for eval_idx, micro in enumerate(tqdm(self.data)):
            latent, vq_loss, usages = self.rec_model(
                img=micro['img_to_encoder'],
                behaviour='enc_dec_wo_triplane')  # pred: (B, 3, 64, 64)
            print(usages)

        for i in range(len(usages)):
            print('usages_for_{}_plane:{}'.format(i, usages[i]))    

class TrainLoop3DRecNVPatchSingleForwardMV(TrainLoop3DRecNVPatchSingleForward):

    def __init__(self,
                 *,
                 rec_model,
                 loss_class,
                 data,
                 eval_data,
                 batch_size,
                 microbatch,
                 lr,
                 ema_rate,
                 log_interval,
                 eval_interval,
                 save_interval,
                 resume_checkpoint,
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
                         loss_class=loss_class,
                         data=data,
                         eval_data=eval_data,
                         batch_size=batch_size,
                         microbatch=microbatch,
                         lr=lr,
                         ema_rate=ema_rate,
                         log_interval=log_interval,
                         eval_interval=eval_interval,
                         save_interval=save_interval,
                         resume_checkpoint=resume_checkpoint,
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         **kwargs)

    def forward_backward(self, batch, behaviour='g_step', *args, **kwargs):
        # Add patch sampling

        if behaviour == 'g_step':
            self.mp_trainer_rec.zero_grad()
        else:
            self.mp_trainer_disc.zero_grad()
            
        batch_size = batch['img_to_encoder'].shape[0]

        batch.pop('caption')  # Not required
        batch.pop('ins')  # Not required

        if '__key__' in batch.keys():
            batch.pop('__key__')

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k:
                v[i:i + self.microbatch].to(dist_util.dev()) if isinstance(
                    v, th.Tensor) else v[i:i + self.microbatch]
                for k, v in batch.items()
            }

            # Sample rendering patch
            nv_c = th.cat([micro['nv_c'], micro['c']])
            target = {
                **self.eg3d_model(
                    c=nv_c,  # type: ignore
                    ws=None,
                    planes=None,
                    sample_ray_only=True,
                    fg_bbox=th.cat([micro['nv_bbox'], micro['bbox']])),  # Rays o / dir
            }

            patch_rendering_resolution = self.eg3d_model.rendering_kwargs[
                'patch_rendering_resolution']  # type: ignore
            cropped_target = {
                k:
                th.empty_like(v).repeat_interleave(2, 0)
                [..., :patch_rendering_resolution, :patch_rendering_resolution]
                if k not in [
                    'ins_idx', 'img_to_encoder', 'img_sr', 'nv_img_to_encoder',
                    'nv_img_sr', 'c', 'caption', 'nv_caption'
                ] else v
                for k, v in micro.items()
            }

            # Crop according to UV sampling
            for j in range(2 * self.microbatch):
                top, left, height, width = target['ray_bboxes'][
                    j]  # List of tuple
                for key in ('img', 'depth_mask', 'depth'):  # type: ignore

                    if j < self.microbatch:
                        cropped_target[f'{key}'][  # No nv_ here
                            j:j + 1] = torchvision.transforms.functional.crop(
                                micro[f'nv_{key}'][j:j + 1], top, left, height,
                                width)
                    else:
                        cropped_target[f'{key}'][  # No nv_ here
                            j:j + 1] = torchvision.transforms.functional.crop(
                                micro[f'{key}'][j - self.microbatch:j -
                                                self.microbatch + 1], top,
                                left, height, width)

            # VIT no AMP
            with th.autocast(device_type='cuda',
                             dtype=self.dtype,
                             enabled=self.mp_trainer_rec.use_amp):

                from .script_util import AE
                AE.forward
                latent, vq_loss, usages = self.rec_model(img=micro['img_to_encoder'].to(self.dtype),
                                        behaviour='enc_dec_wo_triplane')
                ray_origins = target['ray_origins']
                ray_directions = target['ray_directions']
                AE.forward

                pred_nv_cano = self.rec_model(
                    latent={
                        'latent_after_vit':  # Triplane for rendering
                        latent['latent_after_vit'].repeat_interleave(6, dim=0).repeat(2,1,1,1)  # NV=4
                    },
                    c=nv_c,
                    behaviour='triplane_dec',
                    ray_origins=ray_origins,
                    ray_directions=ray_directions,
                )

                pred_nv_cano.update(latent)
                gt = cropped_target

                with self.rec_model.no_sync():  # type: ignore
                    from nsr.losses.builder import E3DGELossClass
                    E3DGELossClass.forward
                    from nsr.losses.builder import E3DGE_with_AdvLoss
                    E3DGE_with_AdvLoss.forward
                    pred_nv_cano.update({'vq_loss': vq_loss})
                    loss, loss_dict, _ = self.loss_class(
                        pred_nv_cano,
                        gt,  # Prepare merged data
                        step=self.step + self.resume_step,
                        test_mode=False,
                        return_fg_mask=True,
                        behaviour=behaviour,
                        conf_sigma_l1=None,
                        conf_sigma_percl=None)
                    for i in range(len(usages)):
                        try:
                            key = f'usage_{i}_plane'
                            values = usages[i]
                            logger.logkv_mean(key, values)
                        except:
                            print('type error:', key)
                    log_rec3d_loss_dict(loss_dict)
            
            if behaviour == 'g_step':
                self.mp_trainer_rec.backward(loss)
            else:
                self.mp_trainer_disc.backward(loss)

            log_vis_interval  = 400
            if dist_util.get_rank() == 0 and self.step % log_vis_interval == 0:
                try:
                    torchvision.utils.save_image(
                        th.cat(
                            [cropped_target['img'], pred_nv_cano['image_raw']
                             ], ),
                        f'{logger.get_dir()}/{self.step+self.resume_step}.jpg',
                        normalize=True, nrow=12)
                    torchvision.utils.save_image(micro['img_to_encoder'][:,:3,:,:],f'{logger.get_dir()}/{self.step+self.resume_step}_input.jpg',normalize=True, nrow=6)
                    
                    logger.log(
                        'log vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')
                except Exception as e:
                    logger.log(e)

class TrainLoop3DRecNVPatchSingleForwardMVAdvLoss(
        TrainLoop3DRecNVPatchSingleForwardMV):

    def __init__(self,
                 *,
                 rec_model,
                 loss_class,
                 data,
                 eval_data,
                 batch_size,
                 microbatch,
                 lr,
                 ema_rate,
                 log_interval,
                 eval_interval,
                 save_interval,
                 resume_checkpoint,
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
                         loss_class=loss_class,
                         data=data,
                         eval_data=eval_data,
                         batch_size=batch_size,
                         microbatch=microbatch,
                         lr=lr,
                         ema_rate=ema_rate,
                         log_interval=log_interval,
                         eval_interval=eval_interval,
                         save_interval=save_interval,
                         resume_checkpoint=resume_checkpoint,
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         **kwargs)

        # Create discriminator
        disc_params = self.loss_class.get_trainable_parameters()
        self.mp_trainer_disc = MixedPrecisionTrainer(
            model=self.loss_class.discriminator,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            model_name='disc',
            use_amp=use_amp,
            model_params=disc_params)
        
        # Resume checkpoint for discriminator
        if self.resume_checkpoint:
            dir_name, file_name = os.path.split(self.resume_checkpoint)
            disc_file_name = file_name.replace('rec', 'disc')
            self.disc_resume_checkpoint = os.path.join(dir_name, disc_file_name)
            if dist_util.get_rank() == 0:
                logger.log(
                    f"Loading model from checkpoint: {self.resume_checkpoint}...")
                map_location = {
                    'cuda:%d' % 0: 'cuda:%d' % dist_util.get_rank()
                }
                disc_resume_state_dict = dist_util.load_state_dict(
                    self.disc_resume_checkpoint, map_location=map_location)
                self.loss_class.discriminator.load_state_dict(disc_resume_state_dict, strict=True)
                logger.log('Discriminator model loading finished')
            
            # Sync parameters
            if dist_util.get_world_size() > 1:
                dist_util.sync_params(self.loss_class.discriminator.parameters())
                logger.log('Synced parameters')

        self.opt_disc = AdamW(
            self.mp_trainer_disc.master_params,
            lr=self.lr,  # Follow sd code base
            betas=(0, 0.999),
            eps=1e-8)

        # Check if loss class is already in the DDP
        if self.use_ddp:
            self.ddp_disc = DDP(
                self.loss_class.discriminator,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
            )
        else:
            self.ddp_disc = self.loss_class.discriminator

    def save(self, mp_trainer=None, model_name='rec'):
        if mp_trainer is None:
            mp_trainer = self.mp_trainer_rec

        def save_checkpoint(rate, params):
            state_dict = mp_trainer.master_params_to_state_dict(params)
            if dist_util.get_rank() == 0:
                logger.log(f"Saving model {model_name} {rate}...")
                if not rate:
                    filename = f"model_{model_name}{(self.step+self.resume_step):07d}.pt"
                else:
                    filename = f"ema_{model_name}_{rate}_{(self.step+self.resume_step):07d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename),
                                 "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, mp_trainer.master_params)

        dist.barrier()

    def run_step(self, batch, step='g_step'):
        if step == 'g_step':
            self.forward_backward(batch, behaviour='g_step')
            took_step_g_rec = self.mp_trainer_rec.optimize(self.opt)

            if took_step_g_rec:
                self._update_ema()  # g_ema

        elif step == 'd_step':
            self.forward_backward(batch, behaviour='d_step')
            _ = self.mp_trainer_disc.optimize(self.opt_disc)

        self._anneal_lr()
        self.log_step()

    def run_loop(self, batch=None):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            batch = next(self.data)
            self.run_step(batch, 'g_step')

            batch = next(self.data)
            self.run_step(batch, 'd_step')

            if self.step % 1000 == 0:
                dist_util.synchronize()
                if self.step % 10000 == 0:
                    th.cuda.empty_cache()  # Avoid memory leak

            if self.step % self.log_interval == 0 and dist_util.get_rank(
            ) == 0:
                out = logger.dumpkvs()
                # Log to tensorboard
                for k, v in out.items():
                    self.writer.add_scalar(f'Loss/{k}', v,
                                           self.step + self.resume_step)
            if self.step % self.eval_interval == 0 and self.step != 0:
                if dist_util.get_rank() == 0:
                    try:
                        self.eval_loop()
                    except Exception as e:
                        logger.log(e)
                dist_util.synchronize()

            if self.step % self.save_interval == 0:
                self.save()
                self.save(self.mp_trainer_disc,
                          self.mp_trainer_disc.model_name)
                dist_util.synchronize()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST",
                                  "") and self.step > 0:
                    return

            self.step += 1

            if self.step > self.iterations:
                logger.log('Reached maximum iterations, exiting')

                # Save the last checkpoint if it wasn't already saved.
                if (self.step -
                        1) % self.save_interval != 0 and self.step != 1:
                    self.save()

                exit()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()  # Save rec
            self.save(self.mp_trainer_disc, self.mp_trainer_disc.model_name)

class TrainLoop3DRecNVPatchSingleForwardMVAdvLossMesh(
        TrainLoop3DRecNVPatchSingleForwardMVAdvLoss):

    def __init__(self,
                 *,
                 rec_model,
                 loss_class,
                 data,
                 eval_data,
                 batch_size,
                 microbatch,
                 lr,
                 ema_rate,
                 log_interval,
                 eval_interval,
                 save_interval,
                 resume_checkpoint,
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 init_model=None,
                 **kwargs):
        super().__init__(rec_model=rec_model,
                         loss_class=loss_class,
                         data=data,
                         eval_data=eval_data,
                         batch_size=batch_size,
                         microbatch=microbatch,
                         lr=lr,
                         ema_rate=ema_rate,
                         log_interval=log_interval,
                         eval_interval=eval_interval,
                         save_interval=save_interval,
                         resume_checkpoint=resume_checkpoint,
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         **kwargs)
        if self.resume_checkpoint == '':
            self.init_model_path = init_model
            if dist_util.get_rank() == 0:
                self.initiate_model(rec_model=self.rec_model, init_model_path=self.init_model_path)
            if dist_util.get_world_size() > 1:
                dist_util.sync_params(self.rec_model.parameters())
                dist_util.sync_params(self.loss_class.discriminator.parameters())
                logger.log('Synced parameters')
                dist.barrier()

        # Check consistency across all GPUs
        rec_consistency = check_model_consistency(self.rec_model, DDP=True)
        if not rec_consistency:
            raise ValueError("Model consistency check failed, exiting.")   
        self.scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opt, 100000, eta_min=0)

    def initiate_model(self, rec_model, init_model_path):
        """
        Instantiate the Mesh model with the pretrained NeRF model.
        """
        sd = th.load(init_model_path, map_location='cpu')
        sd_fc = {}
        for k, v in sd.items():
            if k.startswith('decoder.triplane_decoder.decoder.net.'):
                if k.startswith('decoder.triplane_decoder.decoder.net.6'):
                    if 'weight' in k:
                        sd_fc[k.replace('net.', 'net_sdf.')] = -v[0:1]
                    else:
                        sd_fc[k.replace('net.', 'net_sdf.')] = 10 - v[0:1]
                    sd_fc[k.replace('net.', 'net_rgb.')] = v[1:4]
                else:
                    sd_fc[k.replace('net.', 'net_sdf.')] = v
                    sd_fc[k.replace('net.', 'net_rgb.')] = v
            else:
                sd_fc[k] = v
        # Missing 'net_deformation' and 'net_weight' parameters
        missing_unexpected = rec_model.module.load_state_dict(sd_fc, strict=False)

        # Load checkpoint for discriminator
        if True:
            dir_name, file_name = os.path.split(init_model_path)
            disc_file_name = file_name.replace('rec', 'disc')
            self.disc_resume_checkpoint = os.path.join(dir_name, disc_file_name)
            if dist_util.get_rank() == 0:
                logger.log(
                    f"Loading model from checkpoint: {self.resume_checkpoint}...")
                map_location = {
                    'cuda:%d' % 0: 'cuda:%d' % dist_util.get_rank()
                }
                disc_resume_state_dict = dist_util.load_state_dict(
                    self.disc_resume_checkpoint, map_location=map_location)
                self.loss_class.discriminator.load_state_dict(disc_resume_state_dict, strict=True)
                logger.log('Discriminator model loading finished')

        # Print missing and unexpected keys
        if missing_unexpected.missing_keys:
            print("Missing keys (not found in loaded state_dict):")
            for key in missing_unexpected.missing_keys:
                print(key)

        if missing_unexpected.unexpected_keys:
            print("Unexpected keys (found in loaded state_dict but not in model):")
            for key in missing_unexpected.unexpected_keys:
                print(key)
        logger.info('Init model from {}'.format(init_model_path))

    def run_loop(self, batch=None):
        device = dist_util.dev()
        self.rec_model.module.decoder.triplane_decoder.init_flexicubes_geometry(device=device, fovy=43)
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            batch = next(self.data)
            self.run_step(batch, 'g_step')

            batch = next(self.data)
            self.run_step(batch, 'd_step')

            if self.step % 1000 == 0:
                dist_util.synchronize()
                if self.step % 10000 == 0:
                    th.cuda.empty_cache()  # Avoid memory leak

            if self.step % self.log_interval == 0 and dist_util.get_rank(
            ) == 0:
                try:
                    out = logger.dumpkvs()
                    # Log to tensorboard
                    for k, v in out.items():
                        self.writer.add_scalar(f'Loss/{k}', v,
                                            self.step + self.resume_step)
                except Exception as e:
                    pass
            if self.step % self.eval_interval == 0 and self.step != 0:
                if dist_util.get_rank() == 0:
                    try:
                        self.eval_loop()
                    except Exception as e:
                        logger.log(e)
                dist_util.synchronize()

            if self.step % self.save_interval == 0:
                self.save()
                self.save(self.mp_trainer_disc,
                          self.mp_trainer_disc.model_name)
                dist_util.synchronize()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST",
                                  "") and self.step > 0:
                    return

            self.step += 1

            if self.step > self.iterations:
                logger.log('Reached maximum iterations, exiting')

                # Save the last checkpoint if it wasn't already saved.
                if (self.step -
                        1) % self.save_interval != 0 and self.step != 1:
                    self.save()

                exit()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()  # Save rec
            self.save(self.mp_trainer_disc, self.mp_trainer_disc.model_name)

    def forward_backward(self, batch, behaviour='g_step', *args, **kwargs):
        if behaviour == 'g_step':
            self.mp_trainer_rec.zero_grad()
        elif behaviour == 'd_step':
            self.mp_trainer_disc.zero_grad()
        else:
            raise ValueError('Behaviour not defined')
            
        batch_size = batch['img_to_encoder'].shape[0]
        batch.pop('caption')  # Not required
        batch.pop('ins')  # Not required

        if '__key__' in batch.keys():
            batch.pop('__key__')

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k:
                v[i:i + self.microbatch].to(dist_util.dev()) if isinstance(
                    v, th.Tensor) else v[i:i + self.microbatch]
                for k, v in batch.items()
            }

            nv_c = th.cat([micro['nv_c'], micro['c']])

            patch_rendering_resolution = self.eg3d_model.rendering_kwargs[
                'patch_rendering_resolution']  # type: ignore
            if patch_rendering_resolution != micro['img'].shape[-1]:
                raise ValueError('Patch rendering resolution does not match target image size')
            cropped_target = {
                k:
                th.empty_like(v).repeat_interleave(2, 0)
                [..., :patch_rendering_resolution, :patch_rendering_resolution]
                if k not in [
                    'ins_idx', 'img_to_encoder', 'img_sr', 'nv_img_to_encoder',
                    'nv_img_sr', 'c', 'caption', 'nv_caption'
                ] else v
                for k, v in micro.items()
            }

            for j in range(2 * self.microbatch):
                top, left, height, width = 0, 0, patch_rendering_resolution, patch_rendering_resolution
                for key in ('img', 'depth_mask', 'depth', 'normal'):

                    if j < self.microbatch:
                        cropped_target[f'{key}'][j:j + 1] = torchvision.transforms.functional.crop(
                                micro[f'nv_{key}'][j:j + 1], top, left, height,
                                width)
                    else:
                        cropped_target[f'{key}'][j:j + 1] = torchvision.transforms.functional.crop(
                                micro[f'{key}'][j - self.microbatch:j -
                                                self.microbatch + 1], top,
                                left, height, width)

            with th.autocast(device_type='cuda',
                             dtype=self.dtype,
                             enabled=self.mp_trainer_rec.use_amp):

                from .script_util import AE
                AE.forward
                with th.no_grad():
                    latent, vq_loss, usages = self.rec_model(img=micro['img_to_encoder'].to(self.dtype),
                                            behaviour='enc_dec_wo_triplane')
                AE.forward
                mesh_camera = nv_c.clone()
                mesh_camera = mesh_camera.reshape(-1, 6, 25)
                pred_nv_cano = self.rec_model(
                    latent={
                        'latent_after_vit':  # Triplane for rendering
                        latent['latent_after_vit']
                    },
                    c=mesh_camera,
                    behaviour='triplane_dec',
                    ray_origins=None,
                    ray_directions=None,
                )
                pred_nv_cano.update(latent)
                gt = cropped_target

                with self.rec_model.no_sync():  # type: ignore
                    from nsr.losses.builder import E3DGELossClass
                    E3DGELossClass.forward
                    from nsr.losses.builder import E3DGE_with_AdvLoss
                    E3DGE_with_AdvLoss.forward
                    pred_nv_cano.update({'vq_loss': vq_loss})
                    loss, loss_dict, _ = self.loss_class(
                        pred_nv_cano,
                        gt,  # Prepare merged data
                        step=self.step + self.resume_step,
                        test_mode=False,
                        return_fg_mask=True,
                        behaviour=behaviour,
                        conf_sigma_l1=None,
                        conf_sigma_percl=None)
                    for i in range(len(usages)):
                        try:
                            key = f'usage_{i}_plane'
                            values = usages[i]
                            logger.logkv_mean(key, values)
                        except:
                            print('Type error:', key)
                    log_rec3d_loss_dict(loss_dict)
            
            if behaviour == 'g_step':
                self.mp_trainer_rec.backward(loss)
                self.scheduler.step()
            elif behaviour == 'd_step':
                self.mp_trainer_disc.backward(loss)
            else:
                raise ValueError('Behaviour not defined')

            log_vis_interval  = 100
            if dist_util.get_rank() == 0 and self.step % log_vis_interval == 0:
                try:
                    if behaviour == 'g_step':
                        img_prefix = 'generator'
                    elif behaviour == 'd_step':
                        img_prefix = 'discriminator'
                    torchvision.utils.save_image(
                        th.cat(
                            [cropped_target['img'], pred_nv_cano['image_raw']
                             ], ),
                        f'{logger.get_dir()}/{self.step+self.resume_step}_{img_prefix}.jpg',
                        normalize=True, nrow=12)
                    torchvision.utils.save_image(
                        th.cat(
                            [cropped_target['normal'] * cropped_target['depth_mask'].unsqueeze(1).repeat_interleave(3, 1).float(), pred_nv_cano['image_normal_mesh']
                             ], ),
                        f'{logger.get_dir()}/{self.step+self.resume_step}_normal_{img_prefix}.jpg',
                        normalize=True, nrow=12)
                    torchvision.utils.save_image(micro['img_to_encoder'][:,:3,:,:],f'{logger.get_dir()}/{self.step+self.resume_step}_input_{img_prefix}.jpg',normalize=True, nrow=6)
                    
                    logger.log(
                        'Log vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}_{img_prefix}.jpg')
                except Exception as e:
                    logger.log(e)

    def _init_optim_groups(self, kwargs):
        if kwargs.get('decomposed', False):  # AE
            optim_groups = [
                {
                    'name':
                    'decoder.triplane_decoder',
                    'params':
                    self.mp_trainer_rec.model.decoder.triplane_decoder.parameters(),
                    'lr':
                    kwargs['triplane_decoder_lr'],
                },
            ]

            if self.mp_trainer_rec.model.decoder.decoder_pred_3d is not None:
                optim_groups.append({
                    'name':
                    'decoder_pred_3d',
                    'params':
                    self.mp_trainer_rec.model.decoder.decoder_pred_3d.
                    parameters(),
                    'lr':
                    kwargs['vit_decoder_lr'],
                    'weight_decay':
                    kwargs['vit_decoder_wd']
                })

            if self.mp_trainer_rec.model.decoder.transformer_3D_blk is not None:
                optim_groups.append({
                    'name':
                    'decoder_transformer_3D_blk',
                    'params':
                    self.mp_trainer_rec.model.decoder.transformer_3D_blk.
                    parameters(),
                    'lr':
                    kwargs['vit_decoder_lr'],
                    'weight_decay':
                    kwargs['vit_decoder_wd']
                })

            if self.mp_trainer_rec.model.decoder.logvar is not None:
                optim_groups.append({
                    'name':
                    'decoder_logvar',
                    'params':
                    self.mp_trainer_rec.model.decoder.logvar,
                    'lr':
                    kwargs['vit_decoder_lr'],
                    'weight_decay':
                    kwargs['vit_decoder_wd']
                })

            if self.mp_trainer_rec.model.decoder.decoder_pred is not None:
                optim_groups.append(
                    {
                        'name':
                        'decoder.decoder_pred',
                        'params':
                        self.mp_trainer_rec.model.decoder.decoder_pred.
                        parameters(),
                        'lr':
                        kwargs['vit_decoder_lr'],
                        'weight_decay':
                        kwargs['vit_decoder_wd']
                    }, )

            if self.mp_trainer_rec.model.confnet is not None:
                optim_groups.append({
                    'name':
                    'confnet',
                    'params':
                    self.mp_trainer_rec.model.confnet.parameters(),
                    'lr':
                    1e-5,  # As in unsup3d
                })

            if dist_util.get_rank() == 0:
                logger.log('Using independent optimizer for each component')
        else:
            raise ValueError('Mesh optimization only optimizes the triplane decoder')

        logger.log(optim_groups)

        return optim_groups

class TrainLoop3DRecNVPatchSingleForwardMesh(
        TrainLoop3DRecNVPatchSingleForward):

    def __init__(self,
                 *,
                 rec_model,
                 loss_class,
                 data,
                 eval_data,
                 batch_size,
                 microbatch,
                 lr,
                 ema_rate,
                 log_interval,
                 eval_interval,
                 save_interval,
                 resume_checkpoint,
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 init_model=None,
                 **kwargs):
        super().__init__(rec_model=rec_model,
                         loss_class=loss_class,
                         data=data,
                         eval_data=eval_data,
                         batch_size=batch_size,
                         microbatch=microbatch,
                         lr=lr,
                         ema_rate=ema_rate,
                         log_interval=log_interval,
                         eval_interval=eval_interval,
                         save_interval=save_interval,
                         resume_checkpoint=resume_checkpoint,
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         **kwargs)
        if self.resume_checkpoint == '':
            self.init_model_path = init_model
            if dist_util.get_rank() == 0:
                self.initiate_model(rec_model=self.rec_model, init_model_path=self.init_model_path)
            if dist_util.get_world_size() > 1:
                dist_util.sync_params(self.rec_model.parameters())
                logger.log('Synced parameters')
                dist.barrier()

        # Check consistency across all GPUs
        rec_consistency = check_model_consistency(self.rec_model, DDP=True)
        if not rec_consistency:
            raise ValueError("Model consistency check failed, exiting.")   

    def initiate_model(self, rec_model, init_model_path):
        """
        Instantiate the Mesh model with the pretrained NeRF model.
        """
        sd = th.load(init_model_path, map_location='cpu')
        sd_fc = {}
        for k, v in sd.items():
            if k.startswith('decoder.triplane_decoder.decoder.net.'):
                if k.startswith('decoder.triplane_decoder.decoder.net.6'):
                    if 'weight' in k:
                        sd_fc[k.replace('net.', 'net_sdf.')] = -v[0:1] / 1000
                    else:
                        sd_fc[k.replace('net.', 'net_sdf.')] = -0.065 - v[0:1]
                    sd_fc[k.replace('net.', 'net_rgb.')] = v[1:4]
                else:
                    sd_fc[k.replace('net.', 'net_sdf.')] = v
                    sd_fc[k.replace('net.', 'net_rgb.')] = v
            else:
                sd_fc[k] = v
        # Missing 'net_deformation' and 'net_weight' parameters
        missing_unexpected = rec_model.module.load_state_dict(sd_fc, strict=False)

        # Print missing and unexpected keys
        if missing_unexpected.missing_keys:
            print("Missing keys (not found in loaded state_dict):")
            for key in missing_unexpected.missing_keys:
                print(key)

        if missing_unexpected.unexpected_keys:
            print("Unexpected keys (found in loaded state_dict but not in model):")
            for key in missing_unexpected.unexpected_keys:
                print(key)
        logger.info('Init model from {}'.format(init_model_path))

    def run_loop(self, batch=None):
        device = dist_util.dev()
        self.rec_model.module.decoder.triplane_decoder.init_flexicubes_geometry(device=device, fovy=43)
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            batch = next(self.data)
            self.run_step(batch, 'g_step')

            if self.step % 1000 == 0:
                dist_util.synchronize()
                if self.step % 10000 == 0:
                    th.cuda.empty_cache()  # Avoid memory leak

            if self.step % self.log_interval == 0 and dist_util.get_rank(
            ) == 0:
                out = logger.dumpkvs()
                for k, v in out.items():
                    self.writer.add_scalar(f'Loss/{k}', v,
                                           self.step + self.resume_step)

            if self.step % self.eval_interval == 0 and self.step != 0:
                if dist_util.get_rank() == 0:
                    try:
                        self.eval_loop()
                    except Exception as e:
                        logger.log(e)
                dist_util.synchronize()

            if self.step % self.save_interval == 0:
                self.save()
                dist_util.synchronize()
                if os.environ.get("DIFFUSION_TRAINING_TEST",
                                  "") and self.step > 0:
                    return

            self.step += 1

            if self.step > self.iterations:
                logger.log('Reached maximum iterations, exiting')

                if (self.step -
                        1) % self.save_interval != 0 and self.step != 1:
                    self.save()

                exit()

        if (self.step - 1) % self.save_interval != 0:
            self.save()  # Save rec
            self.save(self.mp_trainer_disc, self.mp_trainer_disc.model_name)

    def forward_backward(self, batch, behaviour='g_step', *args, **kwargs):
        if behaviour == 'g_step':
            self.mp_trainer_rec.zero_grad()
        else:
            raise ValueError('Behaviour not defined')
            
        batch_size = batch['img_to_encoder'].shape[0]
        batch.pop('caption')  # Not required
        batch.pop('ins')  # Not required

        if '__key__' in batch.keys():
            batch.pop('__key__')

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k:
                v[i:i + self.microbatch].to(dist_util.dev()) if isinstance(
                    v, th.Tensor) else v[i:i + self.microbatch]
                for k, v in batch.items()
            }

            nv_c = th.cat([micro['nv_c'], micro['c']])

            patch_rendering_resolution = self.eg3d_model.rendering_kwargs[
                'patch_rendering_resolution']  # type: ignore
            if patch_rendering_resolution != micro['img'].shape[-1]:
                raise ValueError('Patch rendering resolution does not match target image size')
            cropped_target = {
                k:
                th.empty_like(v).repeat_interleave(2, 0)
                [..., :patch_rendering_resolution, :patch_rendering_resolution]
                if k not in [
                    'ins_idx', 'img_to_encoder', 'img_sr', 'nv_img_to_encoder',
                    'nv_img_sr', 'c', 'caption', 'nv_caption'
                ] else v
                for k, v in micro.items()
            }

            for j in range(2 * self.microbatch):
                top, left, height, width = 0, 0, patch_rendering_resolution, patch_rendering_resolution
                for key in ('img', 'depth_mask', 'depth', 'normal'):

                    if j < self.microbatch:
                        cropped_target[f'{key}'][j:j + 1] = torchvision.transforms.functional.crop(
                                micro[f'nv_{key}'][j:j + 1], top, left, height,
                                width)
                    else:
                        cropped_target[f'{key}'][j:j + 1] = torchvision.transforms.functional.crop(
                                micro[f'{key}'][j - self.microbatch:j -
                                                self.microbatch + 1], top,
                                left, height, width)

            with th.autocast(device_type='cuda',
                             dtype=self.dtype,
                             enabled=self.mp_trainer_rec.use_amp):

                from .script_util import AE
                AE.forward
                latent, vq_loss, usages = self.rec_model(img=micro['img_to_encoder'].to(self.dtype),
                                        behaviour='enc_dec_wo_triplane')
                AE.forward
                mesh_camera = nv_c.clone()
                mesh_camera = mesh_camera.reshape(-1, 6, 25)
                pred_nv_cano = self.rec_model(
                    latent={
                        'latent_after_vit':  # Triplane for rendering
                        latent['latent_after_vit']
                    },
                    c=mesh_camera,
                    behaviour='triplane_dec',
                    ray_origins=None,
                    ray_directions=None,
                )
                pred_nv_cano.update(latent)
                gt = cropped_target

                with self.rec_model.no_sync():  # type: ignore
                    from nsr.losses.builder import E3DGELossClass
                    E3DGELossClass.forward
                    from nsr.losses.builder import E3DGE_with_AdvLoss
                    E3DGE_with_AdvLoss.forward
                    pred_nv_cano.update({'vq_loss': vq_loss})
                    loss, loss_dict, _ = self.loss_class(
                        pred_nv_cano,
                        gt,  # Prepare merged data
                        step=self.step + self.resume_step,
                        test_mode=False,
                        return_fg_mask=True,
                        behaviour=behaviour,
                        conf_sigma_l1=None,
                        conf_sigma_percl=None)
                    for i in range(len(usages)):
                        try:
                            key = f'usage_{i}_plane'
                            values = usages[i]
                            logger.logkv_mean(key, values)
                        except:
                            print('Type error:', key)
                    log_rec3d_loss_dict(loss_dict)
            
            if behaviour == 'g_step':
                self.mp_trainer_rec.backward(loss)

            log_vis_interval  = 400
            if dist_util.get_rank() == 0 and self.step % log_vis_interval == 0:
                try:
                    if behaviour == 'g_step':
                        img_prefix = 'generator'
                    torchvision.utils.save_image(
                        th.cat(
                            [cropped_target['img'], pred_nv_cano['image_raw']
                             ], ),
                        f'{logger.get_dir()}/{self.step+self.resume_step}_{img_prefix}.jpg',
                        normalize=True, nrow=12)
                    torchvision.utils.save_image(
                        th.cat(
                            [cropped_target['normal'] * cropped_target['depth_mask'].unsqueeze(1).repeat_interleave(3, 1).float(), pred_nv_cano['image_normal_mesh']
                             ], ),
                        f'{logger.get_dir()}/{self.step+self.resume_step}_normal_{img_prefix}.jpg',
                        normalize=True, nrow=12)
                    torchvision.utils.save_image(micro['img_to_encoder'][:,:3,:,:],f'{logger.get_dir()}/{self.step+self.resume_step}_input_{img_prefix}.jpg',normalize=True, nrow=6)
                    
                    logger.log(
                        'Log vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}_{img_prefix}.jpg')
                except Exception as e:
                    logger.log(e)

def check_model_consistency(model, DDP=True):
    """
    Check if the parameters of the DistributedDataParallel model are consistent across different GPUs.
    """
    if DDP:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    for param_name, param_tensor in state_dict.items():
        tensor_list = [th.zeros_like(param_tensor) for _ in range(th.distributed.get_world_size())]
        
        th.distributed.all_gather(tensor_list, param_tensor)

        for i in range(1, len(tensor_list)):
            if not th.allclose(tensor_list[0], tensor_list[i], rtol=1e-5, atol=1e-8):
                print(f"Parameter '{param_name}' is inconsistent across GPUs.")
                return False

    print("All parameters are consistent across GPUs.")
    return True