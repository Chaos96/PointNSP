import copy
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
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


def flip_yaw(pose_matrix):
    flipped = pose_matrix.clone()
    flipped[:, 0, 1] *= -1
    flipped[:, 0, 2] *= -1
    flipped[:, 1, 0] *= -1
    flipped[:, 2, 0] *= -1
    flipped[:, 0, 3] *= -1
    return flipped


class TrainLoopBasic:

    def __init__(
            self,
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
            fp16_scale_growth=1e-3,
            weight_decay=0.0,
            lr_anneal_steps=0,
            iterations=10001,
            load_submodule_name='',
            ignore_resume_opt=False,
            model_name='rec',
            use_amp=False,
            compile=False,
            **kwargs):
        self.pool_512 = th.nn.AdaptiveAvgPool2d((512, 512))
        self.pool_256 = th.nn.AdaptiveAvgPool2d((256, 256))
        self.pool_128 = th.nn.AdaptiveAvgPool2d((128, 128))
        self.pool_64 = th.nn.AdaptiveAvgPool2d((64, 64))
        self.rec_model = rec_model
        self.loss_class = loss_class
        self.data = data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = ([ema_rate] if isinstance(ema_rate, float) else
                         [float(x) for x in ema_rate.split(",")])
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.iterations = iterations
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist_util.get_world_size()

        self.sync_cuda = th.cuda.is_available()
        self._load_and_sync_parameters()

        self.dtype = th.float32
        if use_amp:
            self.dtype = th.bfloat16

        self.mp_trainer_rec = MixedPrecisionTrainer(
            model=self.rec_model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            model_name=model_name,
            use_amp=use_amp)
        self.writer = SummaryWriter(log_dir=f'{logger.get_dir()}/runs')

        self.opt = AdamW(self._init_optim_groups(kwargs))

        if dist_util.get_rank() == 0:
            logger.log(self.opt)

        if self.resume_step:
            if not ignore_resume_opt:
                self._load_optimizer_state()
            else:
                logger.warn("Ignoring optimizer state from checkpoint.")

            self.ema_params = [
                self._load_ema_parameters(
                    rate,
                    self.rec_model,
                    self.mp_trainer_rec,
                    model_name=self.mp_trainer_rec.model_name)
                for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer_rec.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if compile:
            logger.log('compiling... ignore vit_decoder')
            self.rec_model.decoder.decoder_pred = th.compile(
                self.rec_model.decoder.decoder_pred)
            for module_k, sub_module in self.rec_model.decoder.superresolution.items():
                self.rec_model.decoder.superresolution[module_k] = th.compile(
                    sub_module)

        if th.cuda.is_available():
            self.use_ddp = True
            self.rec_model = th.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.rec_model)

            self.rec_model = DDP(
                self.rec_model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
            )
        else:
            if dist_util.get_world_size() > 1:
                logger.warn("Distributed training requires CUDA. "
                            "Gradients will not be synchronized properly!")
            self.use_ddp = False
            self.rec_model = self.rec_model

        self.novel_view_poses = None
        th.cuda.empty_cache()

    def _init_optim_groups(self, kwargs):
        raise NotImplementedError('')

    def _load_and_sync_parameters(self, submodule_name=''):
        resume_checkpoint = self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(
                resume_checkpoint)
            if dist_util.get_rank() == 0:
                logger.log(
                    f"loading model from checkpoint: {resume_checkpoint}...")
                map_location = {
                    'cuda:%d' % 0: 'cuda:%d' % dist_util.get_rank()
                }
                resume_state_dict = dist_util.load_state_dict(
                    resume_checkpoint, map_location=map_location)

                if submodule_name != '':
                    model_state_dict = getattr(self.rec_model,
                                               submodule_name).state_dict()
                    if dist_util.get_rank() == 0:
                        logger.log('loading submodule: ', submodule_name)
                else:
                    model_state_dict = self.rec_model.state_dict()

                model = self.rec_model


                params_name = []
                for k, v in resume_state_dict.items():
                    if '._orig_mod' in k:
                        k = k.replace('._orig_mod', '')
                    if k in model_state_dict.keys():
                        if v.size() == model_state_dict[k].size():
                            model_state_dict[k] = v
                            params_name.append(k)
                        else:
                            logger.log('!!!! size mismatch, ignore: ', k, ": ",
                                       v.size(), "state_dict: ",
                                       model_state_dict[k].size())
                    elif 'decoder.vit_decoder.blocks' in k:
                        assert len(model.decoder.vit_decoder.blocks[0].vit_blks
                                   ) == 2
                        fusion_ca_depth = len(
                            model.decoder.vit_decoder.blocks[0].vit_blks)
                        vit_subblk_index = int(k.split('.')[3])
                        vit_blk_keyname = ('.').join(k.split('.')[4:])
                        fusion_blk_index = vit_subblk_index // fusion_ca_depth
                        fusion_blk_subindex = vit_subblk_index % fusion_ca_depth
                        model_state_dict[
                            f'decoder.vit_decoder.blocks.{fusion_blk_index}.vit_blks.{fusion_blk_subindex}.{vit_blk_keyname}'] = v
                        logger.log('load 2D ViT weight: {}'.format(
                            f'decoder.vit_decoder.blocks.{fusion_blk_index}.vit_blks.{fusion_blk_subindex}.{vit_blk_keyname}'
                        ))

                    else:
                        logger.log(
                            '!!!! ignore key, not in the model_state_dict: ',
                            k, ": ", v.size())
                self.not_load_params = []
                for k, v in model_state_dict.items():
                    if k not in params_name:
                        self.not_load_params.append(k)
                        logger.log('this key is not loaded: ', k)
                logger.log('model loading finished')

                if submodule_name != '':
                    getattr(self.rec_model,
                            submodule_name).load_state_dict(model_state_dict,
                                                            strict=True)
                else:
                    self.rec_model.load_state_dict(model_state_dict,
                                                   strict=False)

        if dist_util.get_world_size() > 1:
            dist_util.sync_params(self.rec_model.parameters())
            logger.log('synced params')

    def _load_ema_parameters(self,
                             rate,
                             model=None,
                             mp_trainer=None,
                             model_name='ddpm'):

        if mp_trainer is None:
            mp_trainer = self.mp_trainer_rec
        if model is None:
            model = self.rec_model

        ema_params = copy.deepcopy(mp_trainer.master_params)

        main_checkpoint = self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step,
                                             rate, model_name)
        if ema_checkpoint and model_name == 'ddpm':

            if dist_util.get_rank() == 0:

                if not Path(ema_checkpoint).exists():
                    logger.log(
                        f"failed to load EMA from checkpoint: {ema_checkpoint}, not exist"
                    )
                    return

                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")

                map_location = {
                    'cuda:%d' % 0: 'cuda:%d' % dist_util.get_rank()
                }

                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=map_location)

                model_ema_state_dict = model.state_dict()

                for k, v in state_dict.items():
                    if k in model_ema_state_dict.keys() and v.size(
                    ) == model_ema_state_dict[k].size():
                        model_ema_state_dict[k] = v

                    elif 'IN' in k and getattr(model, 'decomposed_IN', False):
                        model_ema_state_dict[k.replace(
                            'IN', 'IN.IN')] = v

                    else:
                        logger.log('ignore key: ', k, ": ", v.size())

                ema_params = mp_trainer.state_dict_to_master_params(
                    model_ema_state_dict)

                del state_dict

        if dist_util.get_world_size() > 1:
            dist_util.sync_params(ema_params)

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint, _ = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint),
                                 f"opt{self.resume_step:06}.pt")
        if bf.exists(opt_checkpoint):
            logger.log(
                f"loading optimizer state from checkpoint: {opt_checkpoint}")

            map_location = {
                'cuda:%d' % 0: 'cuda:%d' % dist_util.get_rank()
            }

            state_dict = dist_util.load_state_dict(opt_checkpoint,
                                                   map_location=map_location)
            self.opt.load_state_dict(state_dict)

            del state_dict

    def run_loop(self, batch=None):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            dist_util.synchronize()

            if isinstance(self.data, list):
                if self.step <= self.data[2]:
                    batch = next(self.data[1])
                else:
                    batch = next(self.data[0])
            else:
                batch = next(self.data)

            if self.novel_view_poses is None:
                self.novel_view_poses = th.roll(batch['c'], 1, 0).to(
                    dist_util.dev())

            self.run_step(batch)

            if self.step % 1000 == 0:
                dist_util.synchronize()
                th.cuda.empty_cache()

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

            if self.step % self.save_interval == 0 and self.step != 0:
                self.save()
                dist_util.synchronize()
                if os.environ.get("DIFFUSION_TRAINING_TEST",
                                  "") and self.step > 0:
                    return

            self.step += 1

            if self.step > self.iterations:
                logger.log('reached maximum iterations, exiting')

                if (self.step -
                        1) % self.save_interval != 0 and self.step != 1:
                    self.save()

                exit()

        if (self.step - 1) % self.save_interval != 0 and self.step != 1:
            self.save()

    @th.no_grad()
    def eval_loop(self):
        raise NotImplementedError('')

    def run_step(self, batch, *args):
        self.forward_backward(batch)
        took_step = self.mp_trainer_rec.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, *args, **kwargs):
        raise NotImplementedError('')

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer_rec.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples",
                     (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):

        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer_rec.master_params_to_state_dict(
                params)
            if dist_util.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model_rec{(self.step+self.resume_step):07d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):07d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename),
                                 "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(
            0, self.mp_trainer_rec.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)
        th.cuda.empty_cache()

        dist.barrier()

class TrainLoop3DRec(TrainLoopBasic):

    def __init__(
            self,
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
            fp16_scale_growth=1e-3,
            weight_decay=0.0,
            lr_anneal_steps=0,
            iterations=10001,
            load_submodule_name='',
            ignore_resume_opt=False,
            model_name='rec',
            use_amp=False,
            compile=False,
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
                         compile=compile,
                         **kwargs)

        self.triplane_scaling_divider = 1.0
        self.latent_name = 'latent_normalized_2Ddiffusion'
        self.render_latent_behaviour = 'decode_after_vae'

        th.cuda.empty_cache()

    @th.inference_mode()
    def render_video_given_triplane(self,
                                    planes,
                                    rec_model,
                                    name_prefix='0',
                                    save_img=False,
                                    render_reference=None,
                                    save_mesh=False):

        planes *= self.triplane_scaling_divider

        batch_size = planes.shape[0]

        ddpm_latent = {'latent_after_vit': planes}

        if save_mesh:
            mesh_size = 192
            mesh_thres = 10
            import mcubes
            import trimesh
            dump_path = f'{logger.get_dir()}/mesh/'

            os.makedirs(dump_path, exist_ok=True)

            grid_out = rec_model(
                latent=ddpm_latent,
                grid_size=mesh_size,
                behaviour='triplane_decode_grid',
            )
            vtx, faces = mcubes.marching_cubes(
                grid_out['sigma'].squeeze(0).squeeze(-1).cpu().numpy(),
                mesh_thres)
            grid_scale = [rec_model.module.decoder.rendering_kwargs['sampler_bbox_min'], rec_model.module.decoder.rendering_kwargs['sampler_bbox_max']]
            vtx = (vtx / (mesh_size-1) * 2 - 1 ) * grid_scale[1]

            vtx_tensor = th.tensor(vtx, dtype=th.float32, device=dist_util.dev()).unsqueeze(0)
            vtx_colors = rec_model.module.decoder.forward_points(ddpm_latent['latent_after_vit'], vtx_tensor)['rgb'].squeeze(0).cpu().numpy()
            vtx_colors = (vtx_colors.clip(0,1) * 255).astype(np.uint8)

            mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=vtx_colors)

            mesh_dump_path = os.path.join(dump_path, f'{name_prefix}.ply')
            mesh.export(mesh_dump_path, 'ply')

            print(f"Mesh dumped to {dump_path}")
            del grid_out, mesh
            th.cuda.empty_cache()

        video_out = imageio.get_writer(
            f'{logger.get_dir()}/triplane_{name_prefix}.mp4',
            mode='I',
            fps=15,
            codec='libx264')

        if render_reference is None:
            render_reference = self.eval_data
        else:
            for key in ['ins', 'bbox', 'caption']:
                if key in render_reference:
                    render_reference.pop(key)

            render_reference = [{
                k: v[idx:idx + 1]
                for k, v in render_reference.items()
            } for idx in range(40)]

        for i, batch in enumerate(tqdm(render_reference)):
            micro = {
                k: v.to(dist_util.dev()) if isinstance(v, th.Tensor) else v
                for k, v in batch.items()
            }
            pred = rec_model(
                latent={
                    'latent_after_vit': ddpm_latent['latent_after_vit'].repeat_interleave(6, dim=0).repeat(2,1,1,1)
                },
                c=micro['c'],
                behaviour='triplane_dec')

            pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())

            pred_depth = pred_depth.cpu()[0].permute(1, 2, 0).numpy()
            pred_depth = (plt.cm.viridis(pred_depth[..., 0])[..., :3]) * 2 - 1
            pred_depth = th.from_numpy(pred_depth).to(
                pred['image_raw'].device).permute(2, 0, 1).unsqueeze(0)

            if 'image_sr' in pred:
                gen_img = pred['image_sr']

                if pred['image_sr'].shape[-1] == 512:
                    pred_vis = th.cat([
                        micro['img_sr'],
                        self.pool_512(pred['image_raw']), gen_img,
                        self.pool_512(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)

                elif pred['image_sr'].shape[-1] == 128:
                    pred_vis = th.cat([
                        micro['img_sr'],
                        self.pool_128(pred['image_raw']), pred['image_sr'],
                        self.pool_128(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)

            else:
                gen_img = pred['image_raw']

                pred_vis = th.cat(
                    [
                        self.pool_128(gen_img),
                        self.pool_128(pred_depth)
                    ],
                    dim=-1)

            if save_img:
                for batch_idx in range(gen_img.shape[0]):
                    sampled_img = Image.fromarray(
                        (gen_img[batch_idx].permute(1, 2, 0).cpu().numpy() *
                         127.5 + 127.5).clip(0, 255).astype(np.uint8))
                    if sampled_img.size != (512, 512):
                        sampled_img = sampled_img.resize(
                            (128, 128), Image.HAMMING)
                    sampled_img.save(logger.get_dir() +
                                     '/FID_Cals/{}_{}.png'.format(
                                         int(name_prefix) * batch_size +
                                         batch_idx, i))

            vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
            vis = vis * 127.5 + 127.5
            vis = vis.clip(0, 255).astype(np.uint8)

            for j in range(vis.shape[0]):
                video_out.append_data(vis[j])

        video_out.close()
        del video_out
        print('logged video to: ',
              f'{logger.get_dir()}/triplane_{name_prefix}.mp4')

        del vis, pred_vis, micro, pred,
    
    @th.inference_mode()
    def render_img_given_triplane(self,
                                    planes,
                                    rec_model,
                                    name_prefix='0',
                                    save_img=False,
                                    render_reference=None,
                                    save_mesh=False,
                                    img=None):
        planes *= self.triplane_scaling_divider

        batch_size = planes.shape[0]

        ddpm_latent = {'latent_after_vit': planes}

        save_mesh = False
        if save_mesh:
            mesh_size = 192
            mesh_thres = 10
            import mcubes
            import trimesh
            dump_path = f'{logger.get_dir()}/mesh/'

            os.makedirs(dump_path, exist_ok=True)

            grid_out = rec_model(
                latent=ddpm_latent,
                grid_size=mesh_size,
                behaviour='triplane_decode_grid',
            )

            vtx, faces = mcubes.marching_cubes(
                grid_out['sigma'].squeeze(0).squeeze(-1).cpu().numpy(),
                mesh_thres)
            grid_scale = [rec_model.module.decoder.rendering_kwargs['sampler_bbox_min'], rec_model.module.decoder.rendering_kwargs['sampler_bbox_max']]
            vtx = (vtx / (mesh_size-1) * 2 - 1 ) * grid_scale[1]

            vtx_tensor = th.tensor(vtx, dtype=th.float32, device=dist_util.dev()).unsqueeze(0)
            vtx_colors = rec_model.module.decoder.forward_points(ddpm_latent['latent_after_vit'], vtx_tensor)['rgb'].squeeze(0).cpu().numpy()
            vtx_colors = (vtx_colors.clip(0,1) * 255).astype(np.uint8)

            mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=vtx_colors)

            mesh_dump_path = os.path.join(dump_path, f'{name_prefix}.ply')
            mesh.export(mesh_dump_path, 'ply')

            print(f"Mesh dumped to {dump_path}")
            del grid_out, mesh
            th.cuda.empty_cache()

        if render_reference is None:
            render_reference = self.eval_data
        else:
            for key in ['ins', 'bbox', 'caption']:
                if key in render_reference:
                    render_reference.pop(key)

            render_reference = [{
                k: v[idx:idx + 1]
                for k, v in render_reference.items()
            } for idx in range(render_reference['c'].shape[0])]

        for i, batch in enumerate(tqdm(render_reference)):
            micro = {
                k: v.to(dist_util.dev()) if isinstance(v, th.Tensor) else v
                for k, v in batch.items()
            }
            pred = rec_model(
                latent={
                    'latent_after_vit': ddpm_latent['latent_after_vit'].repeat_interleave(6, dim=0).repeat(2,1,1,1)
                },
                c=micro['c'],
                behaviour='triplane_dec')

            pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())

            pred_depth = pred_depth.cpu()[0].permute(1, 2, 0).numpy()
            pred_depth = (plt.cm.viridis(pred_depth[..., 0])[..., :3]) * 2 - 1
            pred_depth = th.from_numpy(pred_depth).to(
                pred['image_raw'].device).permute(2, 0, 1).unsqueeze(0)

            if 'image_sr' in pred:
                gen_img = pred['image_sr']

                if pred['image_sr'].shape[-1] == 512:
                    pred_vis = th.cat([
                        micro['img_sr'],
                        self.pool_512(pred['image_raw']), gen_img,
                        self.pool_512(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)

                elif pred['image_sr'].shape[-1] == 128:
                    pred_vis = th.cat([
                        micro['img_sr'],
                        self.pool_128(pred['image_raw']), pred['image_sr'],
                        self.pool_128(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)

            else:
                gen_img = pred['image_raw']

                pred_vis = th.cat(
                    [
                        self.pool_128(gen_img),
                        self.pool_128(pred_depth)
                    ],
                    dim=-1)

            if save_img:
                for batch_idx in range(gen_img.shape[0]):
                    sampled_img = Image.fromarray(
                        (gen_img[batch_idx].permute(1, 2, 0).cpu().numpy() *
                         127.5 + 127.5).clip(0, 255).astype(np.uint8))
                    if sampled_img.size != (512, 512):
                        sampled_img = sampled_img.resize(
                            (128, 128), Image.HAMMING)
                    sampled_img.save(logger.get_dir() +
                                     '/FID_Cals/{}_{}.png'.format(
                                         int(name_prefix) * batch_size +
                                         batch_idx, i))

            comparsion = th.cat([img['img'][i], pred_vis[0][:,:,:128].cpu()], dim=-1)
            torchvision.utils.save_image(comparsion, f'{logger.get_dir()}/triplane_{name_prefix}_vis_{i}.png')

    def get_layer_params(self, model, layer_name):
        layer = model
        for attr in layer_name.split('.'):
            layer = getattr(layer, attr)
        return layer

    def _init_optim_groups(self, kwargs):
        if kwargs.get('decomposed', False):
            vit_encoder_params = []
            vit_decoder_params = []
            decoder_superresolution_params = []

            optim_groups = [
                {
                    'name': 'encoder',
                    'params': self.mp_trainer_rec.model.encoder.parameters(),
                    'lr': kwargs['encoder_lr'],
                    'weight_decay': kwargs['encoder_weight_decay']
                },
                {
                    'name':
                    'decoder.vit_decoder',
                    'params':
                    self.mp_trainer_rec.model.decoder.vit_decoder.parameters(),
                    'lr':
                    kwargs['vit_decoder_lr'],
                    'weight_decay':
                    kwargs['vit_decoder_wd']
                },
                {
                    'name':
                    'decoder.triplane_decoder',
                    'params':
                    self.mp_trainer_rec.model.decoder.triplane_decoder.
                    parameters(),
                    'lr':
                    kwargs['triplane_decoder_lr'],
                },
            ]

            if self.mp_trainer_rec.model.decoder.superresolution is not None:
                optim_groups.append({
                    'name':
                    'decoder.superresolution',
                    'params':
                    self.mp_trainer_rec.model.decoder.superresolution.
                    parameters(),
                    'lr':
                    kwargs['super_resolution_lr'],
                })
            
            if self.mp_trainer_rec.model.dim_up_mlp is not None:
                optim_groups.append({
                    'name':
                    'dim_up_mlp',
                    'params':
                    self.mp_trainer_rec.model.dim_up_mlp.parameters(),
                    'lr':
                    kwargs['encoder_lr'],
                })

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
                    1e-5,
                })

            if dist_util.get_rank() == 0:
                logger.log('using independent optimizer for each components')
        else:
            optim_groups = [
                dict(name='mp_trainer.master_params',
                     params=self.mp_trainer_rec.master_params,
                     lr=self.lr,
                     weight_decay=self.weight_decay)
            ]

        logger.log(optim_groups)

        return optim_groups

    @th.no_grad()
    def eval_novelview_loop(self):
        video_out = imageio.get_writer(
            f'{logger.get_dir()}/video_novelview_{self.step+self.resume_step}.mp4',
            mode='I',
            fps=60,
            codec='libx264')

        all_loss_dict = []
        novel_view_micro = {}

        for i, batch in enumerate(tqdm(self.eval_data)):
            micro = {k: v.to(dist_util.dev()) for k, v in batch.items()}

            if i == 0:
                novel_view_micro = {
                    k:
                    v[0:1].to(dist_util.dev()).repeat_interleave(
                        micro['img'].shape[0], 0)
                    if isinstance(v, th.Tensor) else v[0:1]
                    for k, v in batch.items()
                }
            else:
                novel_view_micro = {
                    k:
                    v[0:1].to(dist_util.dev()).repeat_interleave(
                        micro['img'].shape[0], 0)
                    for k, v in novel_view_micro.items()
                }

            pred = self.rec_model(img=novel_view_micro['img_to_encoder'],
                                  c=micro['c'])

            _, loss_dict = self.loss_class(pred, micro, test_mode=True)
            all_loss_dict.append(loss_dict)

            pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())
            if 'image_sr' in pred:

                if pred['image_sr'].shape[-1] == 512:

                    pred_vis = th.cat([
                        micro['img_sr'],
                        self.pool_512(pred['image_raw']), pred['image_sr'],
                        self.pool_512(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)

                elif pred['image_sr'].shape[-1] == 256:

                    pred_vis = th.cat([
                        micro['img_sr'],
                        self.pool_256(pred['image_raw']), pred['image_sr'],
                        self.pool_256(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)

                else:
                    pred_vis = th.cat([
                        micro['img_sr'],
                        self.pool_128(pred['image_raw']),
                        self.pool_128(pred['image_sr']),
                        self.pool_128(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)

            else:
                pred_vis = th.cat([
                    self.pool_128(micro['img']),
                    self.pool_128(pred['image_raw']),
                    self.pool_128(pred_depth).repeat_interleave(3, dim=1)
                ],
                                  dim=-1)

            vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
            vis = vis * 127.5 + 127.5
            vis = vis.clip(0, 255).astype(np.uint8)

            for j in range(vis.shape[0]):
                video_out.append_data(vis[j])

        video_out.close()

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
    def eval_loop(self):
        video_out = imageio.get_writer(
            f'{logger.get_dir()}/video_{self.step+self.resume_step}.mp4',
            mode='I',
            fps=60,
            codec='libx264')
        all_loss_dict = []
        self.rec_model.eval()

        for i, batch in enumerate(tqdm(self.eval_data)):
            micro = {
                k: v.to(dist_util.dev()) if isinstance(v, th.Tensor) else v
                for k, v in batch.items()
            }

            pred = self.rec_model(img=micro['img_to_encoder'],
                                  c=micro['c'])

            _, loss_dict = self.loss_class(pred, micro, test_mode=True)
            all_loss_dict.append(loss_dict)

            pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())

            if 'image_sr' in pred:

                if pred['image_sr'].shape[-1] == 512:

                    pred_vis = th.cat([
                        micro['img_sr'],
                        self.pool_512(pred['image_raw']), pred['image_sr'],
                        self.pool_512(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)

                elif pred['image_sr'].shape[-1] == 256:
                    pred_vis = th.cat([
                        micro['img_sr'],
                        self.pool_256(pred['image_raw']), pred['image_sr'],
                        self.pool_256(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)

                else:
                    pred_vis = th.cat([
                        micro['img_sr'],
                        self.pool_128(pred['image_raw']),
                        self.pool_128(pred['image_sr']),
                        self.pool_128(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)

            else:
                pred_vis = th.cat([
                    self.pool_128(micro['img']),
                    self.pool_128(pred['image_raw']),
                    self.pool_128(pred_depth).repeat_interleave(3, dim=1)
                ],
                                  dim=-1)

            vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
            vis = vis * 127.5 + 127.5
            vis = vis.clip(0, 255).astype(np.uint8)

            for j in range(vis.shape[0]):
                video_out.append_data(vis[j])

        video_out.close()

        val_scores_for_logging = calc_average_loss(all_loss_dict)
        with open(os.path.join(logger.get_dir(), 'scores.json'), 'a') as f:
            json.dump({'step': self.step, **val_scores_for_logging}, f)

        for k, v in val_scores_for_logging.items():
            self.writer.add_scalar(f'Eval/Rec/{k}', v,
                                   self.step + self.resume_step)

        th.cuda.empty_cache()
        self.eval_novelview_loop()
        self.rec_model.train()

    @th.inference_mode()
    def eval_novelview_loop_trajectory(self):
        for i, batch in enumerate(tqdm(self.eval_data)):
            micro = {k: v.to(dist_util.dev()) for k, v in batch.items()}

            video_out = imageio.get_writer(
                f'{logger.get_dir()}/video_novelview_{self.step+self.resume_step}_batch_{i}.mp4',
                mode='I',
                fps=60,
                codec='libx264')

            for idx, c in enumerate(self.all_nvs_params):
                pred = self.rec_model(img=micro['img_to_encoder'],
                                      c=c.unsqueeze(0).repeat_interleave(
                                          micro['img'].shape[0],
                                          0))

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
                    pred_vis = th.cat([
                        self.pool_128(micro['img']),
                        self.pool_128(pred['image_raw']),
                        self.pool_128(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)

                    pred_vis = pred_vis.permute(0, 2, 3, 1).flatten(0,
                                                                    1)

                vis = pred_vis.cpu().numpy()
                vis = vis * 127.5 + 127.5
                vis = vis.clip(0, 255).astype(np.uint8)

                video_out.append_data(vis)

            video_out.close()

        th.cuda.empty_cache()

    def _prepare_nvs_pose(self):

        device = dist_util.dev()

        fov_deg = 18.837
        intrinsics = FOV_to_intrinsics(fov_deg, device=device)

        all_nvs_params = []

        pitch_range = 0.25
        yaw_range = 0.35
        num_keyframes = 10
        w_frames = 1

        cam_pivot = th.Tensor(
            self.rendering_kwargs.get('avg_camera_pivot')).to(device)
        cam_radius = self.rendering_kwargs.get('avg_camera_radius')

        for frame_idx in range(num_keyframes):

            cam2world_pose = LookAtPoseSampler.sample(
                3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx /
                                              (num_keyframes * w_frames)),
                3.14 / 2 - 0.05 +
                pitch_range * np.cos(2 * 3.14 * frame_idx /
                                     (num_keyframes * w_frames)),
                cam_pivot,
                radius=cam_radius,
                device=device)

            camera_params = th.cat(
                [cam2world_pose.reshape(-1, 16),
                 intrinsics.reshape(-1, 9)], 1)

            all_nvs_params.append(camera_params)

        self.all_nvs_params = th.cat(all_nvs_params, 0)

    def forward_backward(self, batch, *args, **kwargs):
        self.mp_trainer_rec.zero_grad()
        batch_size = batch['img_to_encoder'].shape[0]

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev())
                for k, v in batch.items()
            }

            last_batch = (i + self.microbatch) >= batch_size

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):

                pred = self.rec_model(img=micro['img_to_encoder'],
                                      c=micro['c'])
                target = micro

                conf_sigma_percl = None
                conf_sigma_percl_flip = None
                if 'conf_sigma' in pred:
                    all_conf_sigma_l1 = th.nn.functional.interpolate(
                        pred['conf_sigma'],
                        size=pred['image_raw'].shape[-2:],
                        mode='bilinear'
                    )
                    conf_sigma_l1 = all_conf_sigma_l1[:, :1]
                    conf_sigma_l1_flip = all_conf_sigma_l1[:, 1:]
                else:
                    conf_sigma = None
                    conf_sigma_l1 = None
                    conf_sigma_l1_flip = None

                with self.rec_model.no_sync():
                    loss, loss_dict, fg_mask = self.loss_class(
                        pred,
                        target,
                        step=self.step + self.resume_step,
                        test_mode=False,
                        return_fg_mask=True,
                        conf_sigma_l1=conf_sigma_l1,
                        conf_sigma_percl=conf_sigma_percl)

                if self.loss_class.opt.symmetry_loss:
                    loss_dict['conf_sigma_log'] = conf_sigma_l1.log()
                    pose, intrinsics = micro['c'][:, :16].reshape(
                        -1, 4, 4), micro['c'][:, 16:]
                    flipped_pose = flip_yaw(pose)
                    mirror_c = th.cat(
                        [flipped_pose.reshape(-1, 16), intrinsics], -1)

                    nvs_pred = self.rec_model(latent={
                        k: v
                        for k, v in pred.items() if 'latent' in k
                    },
                                              c=mirror_c,
                                              behaviour='triplane_dec',
                                              return_raw_only=True)

                    nvs_gt = {
                        k: th.flip(target[k], [-1])
                        for k in
                        ['img']
                    }
                    flipped_fg_mask = th.flip(fg_mask, [-1])

                    with self.rec_model.no_sync():
                        loss_symm, loss_dict_symm = self.loss_class.calc_2d_rec_loss(
                            nvs_pred['image_raw'],
                            nvs_gt['img'],
                            flipped_fg_mask,
                            test_mode=False,
                            step=self.step + self.resume_step,
                            conf_sigma_l1=conf_sigma_l1_flip,
                            conf_sigma_percl=conf_sigma_percl_flip)
                        loss += (loss_symm * 1.0)
                        for k, v in loss_dict_symm.items():
                            loss_dict[f'{k}_symm'] = v
                        loss_dict[
                            'flip_conf_sigma_log'] = conf_sigma_l1_flip.log()

                if self.loss_class.opt.density_reg > 0 and self.step % self.loss_class.opt.density_reg_every == 0:

                    initial_coordinates = th.rand(
                        (batch_size, 1000, 3),
                        device=dist_util.dev()) * 2 - 1
                    perturbed_coordinates = initial_coordinates + th.randn_like(
                        initial_coordinates
                    ) * self.loss_class.opt.density_reg_p_dist
                    all_coordinates = th.cat(
                        [initial_coordinates, perturbed_coordinates], dim=1)

                    sigma = self.rec_model(
                        latent=pred['latent'],
                        coordinates=all_coordinates,
                        directions=th.randn_like(all_coordinates),
                        behaviour='triplane_renderer',
                    )['sigma']

                    sigma_initial = sigma[:, :sigma.shape[1] // 2]
                    sigma_perturbed = sigma[:, sigma.shape[1] // 2:]

                    TVloss = th.nn.functional.l1_loss(
                        sigma_initial,
                        sigma_perturbed) * self.loss_class.opt.density_reg

                    loss_dict.update(dict(tv_loss=TVloss))
                    loss += TVloss

            self.mp_trainer_rec.backward(loss)
            log_rec3d_loss_dict(loss_dict)

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                with th.no_grad():

                    def norm_depth(pred_depth):
                        pred_depth = (pred_depth - pred_depth.min()) / (
                            pred_depth.max() - pred_depth.min())
                        return -(pred_depth * 2 - 1)

                    pred_img = pred['image_raw']
                    gt_img = micro['img']

                    if self.loss_class.opt.symmetry_loss:
                        pred_nv_img = nvs_pred
                    else:
                        pred_nv_img = self.rec_model(
                            img=micro['img_to_encoder'],
                            c=self.novel_view_poses)

                    gt_depth = micro['depth']
                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    gt_depth = norm_depth(gt_depth)

                    fg_mask = pred['image_mask'] * 2 - 1
                    nv_fg_mask = pred_nv_img['image_mask'] * 2 - 1
                    if 'image_depth' in pred:
                        pred_depth = norm_depth(pred['image_depth'])
                        pred_nv_depth = norm_depth(pred_nv_img['image_depth'])
                    else:
                        pred_depth = th.zeros_like(gt_depth)
                        pred_nv_depth = th.zeros_like(gt_depth)

                    if 'image_sr' in pred:
                        if pred['image_sr'].shape[-1] == 512:
                            pred_img = th.cat(
                                [self.pool_512(pred_img), pred['image_sr']],
                                dim=-1)
                            gt_img = th.cat(
                                [self.pool_512(micro['img']), micro['img_sr']],
                                dim=-1)
                            pred_depth = self.pool_512(pred_depth)
                            gt_depth = self.pool_512(gt_depth)

                        elif pred['image_sr'].shape[-1] == 256:
                            pred_img = th.cat(
                                [self.pool_256(pred_img), pred['image_sr']],
                                dim=-1)
                            gt_img = th.cat(
                                [self.pool_256(micro['img']), micro['img_sr']],
                                dim=-1)
                            pred_depth = self.pool_256(pred_depth)
                            gt_depth = self.pool_256(gt_depth)

                        else:
                            pred_img = th.cat(
                                [self.pool_128(pred_img), pred['image_sr']],
                                dim=-1)
                            gt_img = th.cat(
                                [self.pool_128(micro['img']), micro['img_sr']],
                                dim=-1)
                            gt_depth = self.pool_128(gt_depth)
                            pred_depth = self.pool_128(pred_depth)
                    else:
                        gt_img = self.pool_128(gt_img)
                        gt_depth = self.pool_128(gt_depth)

                    pred_vis = th.cat([
                        pred_img,
                        pred_depth.repeat_interleave(3, dim=1),
                        fg_mask.repeat_interleave(3, dim=1),
                    ],
                                      dim=-1)

                    if 'conf_sigma' in pred:
                        conf_sigma_l1 = (1 / (conf_sigma_l1 + 1e-7)
                                         ).repeat_interleave(3, dim=1) * 2 - 1
                        pred_vis = th.cat([
                            pred_vis,
                            conf_sigma_l1,
                        ], dim=-1)

                    pred_vis_nv = th.cat([
                        pred_nv_img['image_raw'],
                        pred_nv_depth.repeat_interleave(3, dim=1),
                        nv_fg_mask.repeat_interleave(3, dim=1),
                    ],
                                         dim=-1)

                    if 'conf_sigma' in pred:
                        conf_sigma_for_vis_flip = (
                            1 / (conf_sigma_l1_flip + 1e-7)).repeat_interleave(
                                3, dim=1) * 2 - 1
                        pred_vis_nv = th.cat(
                            [
                                pred_vis_nv,
                                conf_sigma_for_vis_flip,
                            ],
                            dim=-1)

                    pred_vis = th.cat([pred_vis, pred_vis_nv],
                                      dim=-2)

                    gt_vis = th.cat(
                        [
                            gt_img,
                            gt_depth.repeat_interleave(3, dim=1),
                            th.zeros_like(gt_img)
                        ],
                        dim=-1)

                    if 'conf_sigma' in pred:
                        gt_vis = th.cat([gt_vis, fg_mask],
                                        dim=-1)

                    vis = th.cat([gt_vis, pred_vis], dim=-2)
                    vis_tensor = torchvision.utils.make_grid(
                        vis, nrow=vis.shape[-1] // 64)
                    torchvision.utils.save_image(
                        vis_tensor,
                        f'{logger.get_dir()}/{self.step+self.resume_step}.jpg',
                        value_range=(-1, 1),
                        normalize=True)

                    logger.log(
                        'log vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')

            return pred

class TrainLoop3DTriplaneRec(TrainLoop3DRec):

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
                 compile=False,
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
                         compile=compile,
                         **kwargs)

    @th.inference_mode()
    def eval_loop(self):
        # Novel view synthesis given evaluation camera trajectory
        video_out = imageio.get_writer(
            f'{logger.get_dir()}/video_{self.step+self.resume_step}.mp4',
            mode='I',
            fps=60,
            codec='libx264')
        all_loss_dict = []
        self.rec_model.eval()

        device = dist_util.dev()

        # To get intrinsics
        demo_pose = next(self.data)
        intrinsics = demo_pose['c'][0][16:25].to(device)

        video_out = imageio.get_writer(
            f'{logger.get_dir()}/video_{self.step+self.resume_step}.mp4',
            mode='I',
            fps=24,
            bitrate='10M',
            codec='libx264')

        cam_pivot = th.tensor([0, 0, 0], device=dist_util.dev())
        cam_radius = 1.8

        pitch_range = 0.45
        yaw_range = 3.14
        frames = 72

        for pose_idx, (angle_y, angle_p) in enumerate(
                zip([0.2] * 72, np.linspace(-3.14, 3.14, 72))):

            cam2world_pose = LookAtPoseSampler.sample(
                np.pi / 2 + angle_y,
                np.pi / 2 + angle_p,
                cam_pivot,
                radius=cam_radius,
                device=device)

            camera_params = th.cat(
                [cam2world_pose.reshape(-1, 16),
                 intrinsics.reshape(-1, 9)], 1).to(dist_util.dev())

            micro = {'c': camera_params}

            pred = self.rec_model(c=micro['c'])

            # Normalize depth
            pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())

            pred_vis = th.cat([
                self.pool_128(pred['image_raw']),
                self.pool_128(pred_depth).repeat_interleave(3, dim=1)
            ],
                              dim=-1)  # B, 3, H, W

            vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
            vis = vis * 127.5 + 127.5
            vis = vis.clip(0, 255).astype(np.uint8)

            for j in range(vis.shape[0]):
                video_out.append_data(vis[j])

        video_out.close()

        self.rec_model.train()


class TrainLoop3DRecTrajVis(TrainLoop3DRec):

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
        self.rendering_kwargs = self.rec_model.module.decoder.triplane_decoder.rendering_kwargs  # type: ignore
        self._prepare_nvs_pose()  # For eval novelview visualization

    @th.inference_mode()
    def eval_novelview_loop(self):
        # Novel view synthesis given evaluation camera trajectory
        for i, batch in enumerate(tqdm(self.eval_data)):
            micro = {k: v.to(dist_util.dev()) for k, v in batch.items()}

            video_out = imageio.get_writer(
                f'{logger.get_dir()}/video_novelview_{self.step+self.resume_step}_batch_{i}.mp4',
                mode='I',
                fps=60,
                codec='libx264')

            for idx, c in enumerate(self.all_nvs_params):
                pred = self.rec_model(img=micro['img_to_encoder'],
                                      c=c.unsqueeze(0).repeat_interleave(
                                          micro['img'].shape[0],
                                          0))  # pred: (B, 3, 64, 64)

                # Normalize depth
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

                    pred_vis = th.cat([
                        self.pool_128(micro['img']),
                        self.pool_128(pred['image_raw']),
                        self.pool_128(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)  # B, 3, H, W

                    pred_vis = pred_vis.permute(0, 2, 3, 1).flatten(0,
                                                                    1)  # H W 3

                vis = pred_vis.cpu().numpy()
                vis = vis * 127.5 + 127.5
                vis = vis.clip(0, 255).astype(np.uint8)

                video_out.append_data(vis)

            video_out.close()

        th.cuda.empty_cache()

    def _prepare_nvs_pose(self):

        device = dist_util.dev()

        fov_deg = 18.837  # For ffhq/afhq
        intrinsics = FOV_to_intrinsics(fov_deg, device=device)

        all_nvs_params = []

        pitch_range = 0.25
        yaw_range = 0.35
        num_keyframes = 10  # How many nv poses to sample from
        w_frames = 1

        cam_pivot = th.Tensor(
            self.rendering_kwargs.get('avg_camera_pivot')).to(device)
        cam_radius = self.rendering_kwargs.get('avg_camera_radius')

        for frame_idx in range(num_keyframes):

            cam2world_pose = LookAtPoseSampler.sample(
                3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx /
                                              (num_keyframes * w_frames)),
                3.14 / 2 - 0.05 +
                pitch_range * np.cos(2 * 3.14 * frame_idx /
                                     (num_keyframes * w_frames)),
                cam_pivot,
                radius=cam_radius,
                device=device)

            camera_params = th.cat(
                [cam2world_pose.reshape(-1, 16),
                 intrinsics.reshape(-1, 9)], 1)

            all_nvs_params.append(camera_params)

        self.all_nvs_params = th.cat(all_nvs_params, 0)

