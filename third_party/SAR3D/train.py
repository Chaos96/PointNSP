import gc
import os
import sys
import time
import warnings
from functools import partial

import torch

import utils.dist as dist
from utils import arg_util, misc
from datasets.g_buffer_objaverse import load_data_3D_VAR
from utils.misc import auto_resume

# Import xformers components
from xformers.triton import FusedLayerNorm as LayerNorm
from xformers.components.activations import Activation
from xformers.components.feedforward import fused_mlp

import numpy as np
import torchvision

def infinite_loader(loader):
    """Create an infinite iterator from a data loader"""
    while True:
        yield from loader

def build_everything(args: arg_util.Args):
    """Build all components needed for training"""
    
    # Resume from checkpoint if available
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')

    # Create tensorboard logger
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        tb_lg = misc.DistLogger(misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'), verbose=True)
        tb_lg.flush()
    else:
        tb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()
    
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    
    # Build VAE and VAR models
    from models import VAR, VQVAE, build_vae_var_3D_VAR
    num_classes = 1
    vae_local, var_wo_ddp = build_vae_var_3D_VAR(
        device=dist.get_device(), patch_nums=args.patch_nums,
        num_classes=num_classes, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini, args=args
    )

    # Build data loaders
    if not args.local_debug:
        print(f'[build PT data] ...\n')
        
        # Build validation loader
        ld_val = load_data_3D_VAR(
            file_path=args.data_dir,
            batch_size=args.batch_size,
            reso=args.LN3DiffConfig.image_size,
            reso_encoder=args.LN3DiffConfig.image_size_encoder,
            num_workers=args.num_workers,
            load_depth=True,
            preprocess=vae_local.preprocess,
            dataset_size=args.LN3DiffConfig.dataset_size,
            trainer_name=args.LN3DiffConfig.trainer_name,
            use_lmdb=args.LN3DiffConfig.use_lmdb,
            use_wds=args.LN3DiffConfig.use_wds,
            use_lmdb_compressed=args.LN3DiffConfig.use_lmdb_compressed,
            plucker_embedding=args.LN3DiffConfig.plucker_embedding,
            use_chunk=True,
            eval=True,
            load_whole=True,
            text_conditioned=args.text_conditioned,
        )

        # Build training loader 
        ld_train = load_data_3D_VAR(
            file_path=args.data_dir,
            batch_size=args.batch_size,
            reso=args.LN3DiffConfig.image_size,
            reso_encoder=args.LN3DiffConfig.image_size_encoder,
            num_workers=args.num_workers,
            load_depth=True,
            preprocess=vae_local.preprocess,
            dataset_size=args.LN3DiffConfig.dataset_size,
            trainer_name=args.LN3DiffConfig.trainer_name,
            use_lmdb=args.LN3DiffConfig.use_lmdb,
            use_wds=args.LN3DiffConfig.use_wds,
            use_lmdb_compressed=args.LN3DiffConfig.use_lmdb_compressed,
            plucker_embedding=args.LN3DiffConfig.plucker_embedding,
            use_chunk=True,
            load_whole=False,
            text_conditioned=args.text_conditioned,
        )

        [print(line) for line in auto_resume_info]
        print(f'[dataloader multi processing] ...', end='', flush=True)
        stt = time.time()
        iters_train = int( len(ld_train.dataset) / (args.batch_size * dist.get_world_size()))
        print("iters_train", iters_train)

        ld_train = infinite_loader(ld_train)
        ld_train = iter(ld_train)
        print(f'     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)', flush=True, clean=True)

    else:
        num_classes = 1000
        ld_val = ld_train = None
        iters_train = 10
    
    # Build models and optimizer
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models import VAR, VQVAE, build_vae_var_3D_VAR
    from trainer import VARTrainer
    from utils.amp_sc import AmpOptimizer
    from utils.lr_control import filter_params

    # Load pretrained VQVAE weights
    vae_ckpt = args.vqvae_pretrained_path
    dist.barrier()
    
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    
    # Compile models if enabled
    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    var_wo_ddp: VAR = args.compile_model(var_wo_ddp, args.tfast)
    print("Multiple GPUs:", dist.initialized())
    var: DDP = (DDP if dist.initialized() else NullDDP)(var_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    
    print(f'[INIT] VAR model = {var_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.2f}'
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAE', vae_local), ('VAE.enc', vae_local.encoder), ('VAE.dec', vae_local.decoder), ('VAE.quant', vae_local.decoder.superresolution.quantize))]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAR', var_wo_ddp),)]) + '\n\n')

    # Build optimizer
    names, paras, para_groups = filter_params(var_wo_ddp, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })

    opt_clz = {
        'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}\n')
    
    var_optim = AmpOptimizer(
        mixed_precision=args.fp16, optimizer=opt_clz(params=para_groups, **opt_kw), names=names, paras=paras,
        grad_clip=args.tclip, n_gradient_accumulation=args.ac
    )
    del names, paras, para_groups
    
    # Build trainer
    trainer = VARTrainer(
        device=args.device, patch_nums=args.patch_nums, resos=args.resos,
        vae_local=vae_local, var_wo_ddp=var_wo_ddp, var=var,
        var_opt=var_optim, label_smooth=args.ls,
        dino_image_model=None,
        dino_image_processor=None,
    )

    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=True, skip_vae=True)
    del vae_local, var_wo_ddp, var, var_optim

    if args.local_debug:
        rng = torch.Generator('cpu')
        # rng.manual_seed(0)
        rng.manual_seed(3407)
        B = 4
        inp = torch.rand(B, 3, args.data_load_reso, args.data_load_reso)
        label = torch.ones(B, dtype=torch.long)
        
        me = misc.MetricLogger(delimiter='  ')
        trainer.train_step(
            it=0, g_it=0, stepping=True, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prog_si=args.pg0, prog_wp_it=20,
        )
        trainer.load_state_dict(trainer.state_dict())
        trainer.train_step(
            it=99, g_it=599, stepping=True, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prog_si=-1, prog_wp_it=20,
        )
        print({k: meter.global_avg for k, meter in me.meters.items()})
        
        args.dump_log(); tb_lg.flush(); tb_lg.close()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
        exit(0)
    
    dist.barrier()
    
    return (
        tb_lg, trainer, start_ep, start_it,
        iters_train, ld_train, ld_val
    )

def main_training():
    """Main training loop for SAR3D model.
    
    Key steps:
    1. Initialize model, optimizer, trainer
    2. Train for specified epochs
    3. Evaluate and save checkpoints periodically
    4. Log metrics and visualizations
    """
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)
    
    # Build model and training components
    (
        tb_lg, trainer,
        start_ep, start_it,
        iters_train, ld_train, ld_val
    ) = build_everything(args)
    
    # Initialize training metrics
    start_time = time.time()
    best_metrics = {
        'L_mean': 999., 'L_tail': 999.,
        'acc_mean': -1., 'acc_tail': -1.,
        'val_loss_mean': 999, 'val_loss_tail': 999,
        'val_acc_mean': -1, 'val_acc_tail': -1
    }
    
    L_mean, L_tail = -1, -1
    
    # Main training loop
    for ep in range(start_ep, args.ep):
        # Set epoch for distributed sampler
        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]', flush=True, force=True)
        
        tb_lg.set_step(ep * iters_train)

        # Train one epoch
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep, ep == start_ep, start_it if ep == start_ep else 0, 
            args, tb_lg, ld_train, iters_train, trainer
        )
        
        # Update metrics
        L_mean, L_tail = stats['Lm'], stats['Lt']
        acc_mean, acc_tail = stats['Accm'], stats['Acct']
        grad_norm = stats['tnm']
        
        best_metrics.update({
            'L_mean': min(best_metrics['L_mean'], L_mean),
            'acc_mean': max(best_metrics['acc_mean'], acc_mean)
        })
        
        if L_tail != -1:
            best_metrics.update({
                'L_tail': min(best_metrics['L_tail'], L_tail),
                'acc_tail': max(best_metrics['acc_tail'], acc_tail)
            })
            
        # Update args with current metrics
        args.L_mean, args.L_tail = L_mean, L_tail
        args.acc_mean, args.acc_tail = acc_mean, acc_tail
        args.grad_norm = grad_norm
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        
        # Log epoch metrics
        AR_ep_loss = {
            'L_mean': L_mean, 'L_tail': L_tail,
            'acc_mean': acc_mean, 'acc_tail': acc_tail
        }

        # Save checkpoint and visualize results
        if True:  # Save every epoch for now
            if dist.is_local_master():
                # Save checkpoint
                local_out_ckpt = os.path.join(args.local_out_dir_path, f'ar-ckpt-{ep}.pth')
                local_out_ckpt_best = os.path.join(args.local_out_dir_path, 'ar-ckpt-best.pth')
                
                print(f'[saving ckpt] ...', end='', flush=True)
                torch.save({
                    'epoch': ep+1,
                    'iter': (ep+1) * iters_train,
                    'trainer': trainer.state_dict(),
                    'args': args.state_dict(),
                }, local_out_ckpt)
                
                print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True, clean=True)
                
                # Generate and save visualizations
                if args.text_conditioned:
                    triplane, caption, data_eval = trainer.eval_ep_3D_VAR_text(ld_val)
                else:
                    triplane, caption, data_eval = trainer.eval_ep_3D_VAR(ld_val)
                save_img = []
                
                
                # Render images from triplane representation
                with torch.inference_mode():
                    for i in range(min(triplane.shape[0],4)):
                        triplane_i = triplane[i].unsqueeze(0)
                        camera = data_eval['nv_c'][12*i:12*(i+1)].to(dist.get_rank())
                        # Generate predictions
                        image_raw = []
                        for j in range(camera.shape[0]):
                            camera_j = camera[j].unsqueeze(0)
                            pred = trainer.vae_local(
                                latent={
                                    'latent_after_vit': triplane_i.repeat_interleave(6, dim=0).to(torch.float32).repeat(2,1,1,1)
                                },
                                c=camera_j,
                                behaviour='triplane_dec'
                            )
                            image_raw.append(pred['image_raw'])
                        pred['image_raw'] = torch.cat(image_raw, dim=0)
                        # Combine with ground truth for visualization
                        gt_img = data_eval['nv_img'][12*i:12*(i+1)]
                        save_img_i = torch.cat([pred['image_raw'][0:6], gt_img[0:6].to(pred['image_raw'].device)], dim=2)
                        save_img.append(save_img_i)
                
                save_img = torch.cat(save_img, dim=2)
                
                # Save rendered images
                os.makedirs(os.path.join(args.local_out_dir_path, 'rendered'), exist_ok=True)
                save_path = os.path.join(args.local_out_dir_path, 'rendered', f'ep_{ep}.png')
                
                save_img = (save_img + 1) / 2
                torchvision.utils.save_image(save_img, save_path, nrow=6)

                # Cleanup
                del triplane, caption, data_eval, save_img, camera, pred, gt_img, save_img_i
                del local_out_ckpt, local_out_ckpt_best
                
            dist.barrier()
        
        # Print training progress
        print(f'     [ep{ep}]  (training )  Lm: {best_metrics["L_mean"]:.3f} ({L_mean:.3f}), Lt: {best_metrics["L_tail"]:.3f} ({L_tail:.3f}),  Acc m&t: {best_metrics["acc_mean"]:.2f} {best_metrics["acc_tail"]:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        
        # Log to tensorboard
        tb_lg.update(head='AR_ep_loss', step=ep+1, **AR_ep_loss)
        tb_lg.update(head='AR_z_burnout', step=ep+1, rest_hours=round(sec / 60 / 60, 2))
        args.dump_log()
        tb_lg.flush()
    
    # Print final training summary
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [Training finished]  Total cost: {total_time},   Lm: {best_metrics["L_mean"]:.3f} ({L_mean}),   Lt: {best_metrics["L_tail"]:.3f} ({L_tail})')
    print('\n\n')
    
    # Cleanup
    del stats, iters_train, ld_train
    time.sleep(3)
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)
    
    args.remain_time = '-'
    args.finish_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log()
    tb_lg.flush()
    tb_lg.close()
    dist.barrier()


def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, tb_lg: misc.TensorboardLogger, ld_or_itrt, iters_train: int, trainer):
    """Train model for one epoch.
    
    Args:
        ep: Current epoch number
        is_first_ep: Whether this is the first epoch
        start_it: Starting iteration number
        args: Training arguments
        tb_lg: Tensorboard logger
        ld_or_itrt: Data loader
        iters_train: Number of iterations per epoch
        trainer: Model trainer
        
    Returns:
        Tuple of (metrics dict, timing info)
    """
    from trainer import VARTrainer
    from utils.lr_control import lr_wd_annealing
    trainer: VARTrainer
    
    step_cnt = 0
    
    # Setup metric logging
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
    me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        
    g_it, max_it = ep * iters_train, args.ep * iters_train
    
    # Load empty embeddings for classifier-free guidance
    if args.text_conditioned:
        empty_text_embedding = torch.from_numpy(np.load("./files/empty_text_embedding.npy")).to(dist.get_device()).unsqueeze(0)
        empty_text_pooler_output = torch.from_numpy(np.load("./files/empty_text_pooler_output.npy")).to(dist.get_device()).unsqueeze(0)
    else:
        empty_pooler_output = torch.from_numpy(np.load("./files/empty_dino_pooler_output.npy")).to(dist.get_device()).unsqueeze(0)
        empty_dino_image_embedding = torch.from_numpy(np.load("./files/empty_dino_embedding.npy"))[1:, :].to(dist.get_device()).unsqueeze(0)

    # Training loop
    for it, (data) in me.log_every(start_it, iters_train, ld_or_itrt, 30 if iters_train > 8000 else 5, header):
        label = torch.ones(args.batch_size, dtype=torch.int64)
        
        g_it = ep * iters_train + it
        if it < start_it: continue
        if is_first_ep and it == start_it: warnings.resetwarnings()
        
        label = label.to(args.device, non_blocking=True)
        args.cur_it = f'{it+1}/{iters_train}'
        
        # Learning rate scheduling
        wp_it = args.wp * iters_train
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(
            args.sche, trainer.var_opt.optimizer, args.tlr, 
            args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
        args.cur_lr, args.cur_wd = max_tlr, max_twd
        
        # Progressive training (if enabled)
        prog_si = -1
        if args.pg:
            if g_it <= wp_it: 
                prog_si = args.pg0
            elif g_it >= max_it*args.pg:
                prog_si = len(args.patch_nums) - 1
            else:
                delta = len(args.patch_nums) - 1 - args.pg0
                progress = min(max((g_it - wp_it) / (max_it*args.pg - wp_it), 0), 1)
                prog_si = args.pg0 + round(progress * delta)
                
        stepping = (g_it + 1) % args.ac == 0
        step_cnt += int(stepping)

        # Training step
        if args.text_conditioned:
            grad_norm, scale_log2 = trainer.train_step_text(
                it=it, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,
                inp_B3HW=data, label_B=label, prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
                empty_text_pooler_output=empty_text_pooler_output,
                empty_text_embedding=empty_text_embedding,
            )
        
        else:
            grad_norm, scale_log2 = trainer.train_step(
                it=it, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,
                inp_B3HW=data, label_B=label, prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
                empty_pooler_output=empty_pooler_output,
                empty_dino_image_embedding=empty_dino_image_embedding,
            )
        
        # Log metrics
        me.update(tlr=max_tlr)
        tb_lg.set_step(step=g_it)
        tb_lg.update(head='AR_opt_lr/lr_min', sche_tlr=min_tlr)
        tb_lg.update(head='AR_opt_lr/lr_max', sche_tlr=max_tlr)
        tb_lg.update(head='AR_opt_wd/wd_max', sche_twd=max_twd)
        tb_lg.update(head='AR_opt_wd/wd_min', sche_twd=min_twd)
        tb_lg.update(head='AR_opt_grad/fp16', scale_log2=scale_log2)
        
        if args.tclip > 0:
            tb_lg.update(head='AR_opt_grad/grad', grad_norm=grad_norm)
            tb_lg.update(head='AR_opt_grad/grad', grad_clip=args.tclip)
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)


class NullDDP(torch.nn.Module):
    """Dummy DDP wrapper for single GPU training."""
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == '__main__':
    try:
        main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
