"""
Train a diffusion model on images.
"""
import random
import json
import sys
import os

sys.path.append('.')
import torch.distributed as dist
import traceback
import torch as th
import torch.multiprocessing as mp
import numpy as np
import argparse
import dnnlib
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)
from nsr.train_nv_util import TrainLoop3DRecNVPatchSingleForwardMVAdvLoss
from nsr.script_util import create_3DAE_model, encoder_and_nsr_defaults, loss_defaults, rendering_options_defaults, eg3d_options_default, dataset_defaults
from nsr.script_util import create_3DAE_model_mesh
from nsr.losses.builder import E3DGELossClass, E3DGE_with_AdvLoss

enable_tf32 = th.backends.cuda.matmul.allow_tf32 

th.backends.cuda.matmul.allow_tf32 = enable_tf32
th.backends.cudnn.allow_tf32 = enable_tf32
th.backends.cudnn.enabled = True


def training_loop(args):
    dist_util.setup_dist(args)
    th.autograd.set_detect_anomaly(False)  # type: ignore

    SEED = args.seed

    logger.log(f"Local rank: {args.local_rank} initialization complete, seed={SEED}")
    th.cuda.set_device(args.local_rank)
    th.cuda.empty_cache()

    th.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    logger.configure(dir=args.logdir)

    logger.log("Creating encoder and NSR decoder...")
    device = th.device("cuda", args.local_rank)

    opts = eg3d_options_default()

    if args.sr_training:
        args.sr_kwargs = dnnlib.EasyDict(
            channel_base=opts.cbase,
            channel_max=opts.cmax,
            fused_modconv_default='inference_only',
            use_noise=True
        )

    if 'mesh' in args.trainer_name:
        auto_encoder = create_3DAE_model_mesh(
            **args_to_dict(args, encoder_and_nsr_defaults().keys()))
    else:
        auto_encoder = create_3DAE_model(
            **args_to_dict(args, encoder_and_nsr_defaults().keys()))
    auto_encoder.to(device)
    auto_encoder.train()

    logger.log("Creating data loader...")
    if args.objv_dataset:
        from datasets.g_buffer_objaverse import load_data
    else:
        from datasets.shapenet import load_data, load_eval_data, load_memory_data

    if args.overfitting:
        data = load_memory_data(
            file_path=args.data_dir,
            batch_size=args.batch_size,
            reso=args.image_size,
            reso_encoder=args.image_size_encoder,
            num_workers=args.num_workers,
            **args_to_dict(args, dataset_defaults().keys()))
        eval_data = None
    else:
        if args.use_wds:
            if args.data_dir == 'NONE':
                with open(args.shards_lst) as f:
                    shards_lst = [url.strip() for url in f.readlines()]
                data = load_wds_data(
                    shards_lst,
                    args.image_size,
                    args.image_size_encoder,
                    args.batch_size,
                    args.num_workers,
                    **args_to_dict(args, dataset_defaults().keys()))
            elif not args.inference:
                data = load_wds_data(args.data_dir,
                                     args.image_size,
                                     args.image_size_encoder,
                                     args.batch_size,
                                     args.num_workers,
                                     plucker_embedding=args.plucker_embedding,
                                     mv_input=args.mv_input,
                                     split_chunk_input=args.split_chunk_input)
            else:
                data = None

            if args.eval_data_dir == 'NONE':
                with open(args.eval_shards_lst) as f:
                    eval_shards_lst = [url.strip() for url in f.readlines()]
            else:
                eval_shards_lst = args.eval_data_dir

            eval_data = load_wds_data(
                eval_shards_lst,
                args.image_size,
                args.image_size_encoder,
                args.eval_batch_size,
                args.num_workers,
                **args_to_dict(args, dataset_defaults().keys()))
        else:
            eval_data = load_data(
                file_path=args.data_dir,
                batch_size=args.batch_size,
                reso=args.image_size,
                reso_encoder=args.image_size_encoder,
                num_workers=args.num_workers,
                load_depth=True,
                preprocess=auto_encoder.preprocess,
                dataset_size=args.dataset_size,
                trainer_name=args.trainer_name,
                use_lmdb=args.use_lmdb,
                use_wds=args.use_wds,
                use_lmdb_compressed=args.use_lmdb_compressed,
                plucker_embedding=args.plucker_embedding,
                use_chunk=True,
                eval=True
            )
            data = load_data(
                file_path=args.data_dir,
                batch_size=args.batch_size,
                reso=args.image_size,
                reso_encoder=args.image_size_encoder,
                num_workers=args.num_workers,
                load_depth=True,
                preprocess=auto_encoder.preprocess,
                dataset_size=args.dataset_size,
                trainer_name=args.trainer_name,
                use_lmdb=args.use_lmdb,
                use_wds=args.use_wds,
                use_lmdb_compressed=args.use_lmdb_compressed,
                plucker_embedding=args.plucker_embedding,
                use_chunk=True
            )
            if args.pose_warm_up_iter > 0:
                overfitting_dataset = load_memory_data(
                    file_path=args.data_dir,
                    batch_size=args.batch_size,
                    reso=args.image_size,
                    reso_encoder=args.image_size_encoder,
                    num_workers=args.num_workers,
                    **args_to_dict(args, dataset_defaults().keys()))
                data = [data, overfitting_dataset, args.pose_warm_up_iter]

    logger.log("Data loader creation complete.")

    args.img_size = [args.image_size_encoder]

    dist_util.synchronize()

    opt = dnnlib.EasyDict(args_to_dict(args, loss_defaults().keys()))

    if 'disc' in args.trainer_name:
        loss_class = E3DGE_with_AdvLoss(
            device,
            opt,
            disc_factor=args.patchgan_disc_factor,
            disc_weight=args.patchgan_disc_g_weight).to(device)
    else:
        loss_class = E3DGELossClass(device, opt).to(device)

    logger.log("Starting training...")

    TrainLoop = {
        'nv_rec_patch_mvE_disc': TrainLoop3DRecNVPatchSingleForwardMVAdvLoss,
    }[args.trainer_name]

    logger.log("TrainLoop creation complete.")

    auto_encoder.decoder.rendering_kwargs = args.rendering_kwargs
    train_loop = TrainLoop(
        rec_model=auto_encoder,
        loss_class=loss_class,
        data=data,
        eval_data=eval_data,
        **vars(args))

    if args.inference:
        camera = th.load('assets/objv_eval_pose.pt', map_location=dist_util.dev())
        rot = th.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],
                         ], dtype=th.float32, device=camera.device)
        for i in range(camera.shape[0]):
            camera[i][0:16] = th.mm(rot, camera[i][0:16].reshape(4, 4)).reshape(-1)

        train_loop.eval_novelview_loop(camera=camera, save_latent=args.save_latent)
    else:
        train_loop.run_loop()


def create_argparser(**kwargs):
    defaults = dict(
        seed=0,
        dataset_size=-1,
        trainer_name='input_rec',
        use_amp=False,
        overfitting=False,
        num_workers=4,
        image_size=128,
        image_size_encoder=224,
        iterations=150000,
        anneal_lr=False,
        lr=5e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        eval_batch_size=12,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=50,
        eval_interval=2500,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        data_dir="",
        eval_data_dir="",
        logdir="/mnt/lustre/yslan/logs/nips23/",
        pose_warm_up_iter=-1,
        inference=False,
        export_latent=False,
        save_latent=False,
        init_model="",
        grid_res=0.,
        grid_scale=0.,
    )

    defaults.update(dataset_defaults())
    defaults.update(encoder_and_nsr_defaults())
    defaults.update(loss_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    th.multiprocessing.set_start_method('spawn')
    args = create_argparser().parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.gpus = th.cuda.device_count()

    opts = args

    args.rendering_kwargs = rendering_options_defaults(opts)

    with open(os.path.join(args.logdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print('Launching processes...')

    try:
        training_loop(args)
    except Exception as e:
        traceback.print_exc()
        dist_util.cleanup()
