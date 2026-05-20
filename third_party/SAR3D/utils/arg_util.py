import json
import os
import random
import re
import subprocess
import sys
import time
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import torch

try:
    from tap import Tap
except ImportError as e:
    print(f'`>>>>>>>> from tap import Tap` failed, please run:      pip3 install typed-argument-parser     <<<<<<<<', file=sys.stderr, flush=True)
    print(f'`>>>>>>>> from tap import Tap` failed, please run:      pip3 install typed-argument-parser     <<<<<<<<', file=sys.stderr, flush=True)
    time.sleep(5)
    raise e

import utils.dist as dist

class Dict_to_class:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class Args(Tap):
    data_path: str = '/path/to/imagenet'
    exp_name: str = 'text'
    test_image_path: str = None
    text_conditioned: bool = False
    text_json_path: str = None
    
    # VAE
    vfast: int = 0      # torch.compile VAE; =0: not compile; 1: compile with 'reduce-overhead'; 2: compile with 'max-autotune'
    # VAR
    tfast: int = 0      # torch.compile VAR; =0: not compile; 1: compile with 'reduce-overhead'; 2: compile with 'max-autotune'
    depth: int = 16     # VAR depth
    # VAR initialization
    ini: float = -1     # -1: automated model parameter initialization
    hd: float = 0.02    # head.w *= hd
    aln: float = 0.5    # the multiplier of ada_lin.w's initialization
    alng: float = 1e-5  # the multiplier of ada_lin.w[gamma channels]'s initialization
    # VAR optimization
    fp16: int = 0           # 1: using fp16, 2: bf16
    # tblr: float = 1e-4      # base lr
    tblr: float = 1e-4      # base lr
    tlr: float = None       # lr = base lr * (bs / 256)
    twd: float = 0.05       # initial wd
    twde: float = 0         # final wd, =twde or twd
    tclip: float = 2.       # <=0 for not using grad clip
    ls: float = 0.0         # label smooth
    
    bs: int = 768           # global batch size
    batch_size: int = 0     # [automatically set; don't specify this] batch size per GPU = round(args.bs / args.ac / dist.get_world_size() / 8) * 8
    glb_batch_size: int = 0 # [automatically set; don't specify this] global batch size = args.batch_size * dist.get_world_size()
    ac: int = 1             # gradient accumulation
    
    ep: int = 250
    wp: float = 0
    wp0: float = 0.005      # initial lr ratio at the begging of lr warm up
    wpe: float = 0.01       # final lr ratio at the end of training
    sche: str = 'lin0'      # lr schedule
    
    opt: str = 'adamw'      # lion: https://cloud.tencent.com/developer/article/2336657?areaId=106001 lr=5e-5 (0.25x) wd=0.8 (8x); Lion needs a large bs to work
    afuse: bool = True      # fused adamw
    
    # other hps
    saln: bool = False      # whether to use shared adaln
    anorm: bool = True      # whether to use L2 normalized attention
    fuse: bool = True       # whether to use fused op like flash attn, xformers, fused MLP, fused LayerNorm, etc.
    
    # data
    pn: str = '1_2_3_4_5_6_8_10_13_16'
    patch_size: int = 16
    patch_nums: tuple = None    # [automatically set; don't specify this] = tuple(map(int, args.pn.replace('-', '_').split('_')))
    resos: tuple = None         # [automatically set; don't specify this] = tuple(pn * args.patch_size for pn in args.patch_nums)
    
    data_load_reso: int = None  # [automatically set; don't specify this] would be max(patch_nums) * patch_size
    mid_reso: float = 1.125     # aug: first resize to mid_reso = 1.125 * data_load_reso, then crop to data_load_reso
    hflip: bool = False         # augmentation: horizontal flip
    num_workers: int = 0        # num workers; 0: auto, -1: don't use multiprocessing in DataLoader
    
    # progressive training
    pg: float = 0.0         # >0 for use progressive training during [0%, this] of training
    pg0: int = 4            # progressive initial stage, 0: from the 1st token map, 1: from the 2nd token map, etc
    pgwp: float = 0         # num of warmup epochs at each progressive stage
    
    # would be automatically set in runtime
    cmd: str = ' '.join(sys.argv[1:])  # [automatically set; don't specify this]
    branch: str = subprocess.check_output(f'git symbolic-ref --short HEAD 2>/dev/null || git rev-parse HEAD', shell=True).decode('utf-8').strip() or '[unknown]' # [automatically set; don't specify this]
    commit_id: str = subprocess.check_output(f'git rev-parse HEAD', shell=True).decode('utf-8').strip() or '[unknown]'  # [automatically set; don't specify this]
    commit_msg: str = (subprocess.check_output(f'git log -1', shell=True).decode('utf-8').strip().splitlines() or ['[unknown]'])[-1].strip()    # [automatically set; don't specify this]
    acc_mean: float = None      # [automatically set; don't specify this]
    acc_tail: float = None      # [automatically set; don't specify this]
    L_mean: float = None        # [automatically set; don't specify this]
    L_tail: float = None        # [automatically set; don't specify this]
    vacc_mean: float = None     # [automatically set; don't specify this]
    vacc_tail: float = None     # [automatically set; don't specify this]
    vL_mean: float = None       # [automatically set; don't specify this]
    vL_tail: float = None       # [automatically set; don't specify this]
    grad_norm: float = None     # [automatically set; don't specify this]
    cur_lr: float = None        # [automatically set; don't specify this]
    cur_wd: float = None        # [automatically set; don't specify this]
    cur_it: str = ''            # [automatically set; don't specify this]
    cur_ep: str = ''            # [automatically set; don't specify this]
    remain_time: str = ''       # [automatically set; don't specify this]
    finish_time: str = ''       # [automatically set; don't specify this]
    
    # environment
    local_out_dir_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_output')  # [automatically set; don't specify this]
    tb_log_dir_path: str = '...tb-...'  # [automatically set; don't specify this]
    log_txt_path: str = '...'           # [automatically set; don't specify this]
    last_ckpt_path: str = '...'         # [automatically set; don't specify this]
    
    tf32: bool = True       # whether to use TensorFloat32
    device: str = 'cpu'     # [automatically set; don't specify this]
    seed: int = None        # seed
    def seed_everything(self, benchmark: bool):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = benchmark
        if self.seed is None:
            torch.backends.cudnn.deterministic = False
        else:
            torch.backends.cudnn.deterministic = True
            seed = self.seed * dist.get_world_size() + dist.get_rank()
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
    same_seed_for_all_ranks: int = 0     # this is only for distributed sampler
    def get_different_generator_for_each_rank(self) -> Optional[torch.Generator]:   # for random augmentation
        if self.seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(self.seed * dist.get_world_size() + dist.get_rank())
        return g
    
    local_debug: bool = 'KEVIN_LOCAL' in os.environ
    dbg_nan: bool = False   # 'KEVIN_LOCAL' in os.environ

    # use flexicubes to extract mesh and render
    flexicubes: bool = False
    save_path: str = '.sample_data'
    save_BL: bool = False

    # LN3Diff args (TODO: clean these args)
    LN3Diff_kwargs = {
        'image_size_encoder': 256,
        'dino_version': 'mv-sd-dit-dynaInp-trilatent',
        'sr_training': False,
        'cls_token': False,
        'weight_decay': 0.05,
        'image_size': 512,
        'no_dim_up_mlp': True,
        'uvit_skip_encoder': False,
        'fg_mse': True,
        'bg_lamdba': 1.0,
        'lpips_delay_iter': 100,
        'sr_delay_iter': 25000,
        'kl_anneal': True,
        'symmetry_loss': False,
        'vae_p': 2,
        'plucker_embedding': True,
        'encoder_in_channels': 10,
        'arch_dit_decoder': 'DiT2-B/2',
        'sd_E_ch': 64,
        'sd_E_num_res_blocks': 1,
        'lrm_decoder': False,
        'trainer_name': 'nv_rec_patch_mvE',
        'overfitting': False,
        'load_pretrain_encoder': False,
        'decomposed': True,
        'cfg': 'objverse_tuneray_aug_resolution_256_256_auto',
        'patch_size': 14,
        'use_amp': True,
        'eval_batch_size': 1,
        'depth_smoothness_lambda': 0,
        'use_conf_map': False,
        'objv_dataset': True,
        'depth_lambda': 0.5,
        'patch_rendering_resolution': 48,
        'use_lmdb_compressed': False,
        'use_lmdb': False,
        'mv_input': True,
        'split_chunk_input': True,
        'append_depth': True,
        'patchgan_disc_factor': 0.1,
        'patchgan_disc_g_weight': 0.02,
        'use_wds': False,
        'img_size': None,
        'in_chans': 384,
        'num_classes': 0,
        'embed_dim': 384,
        'num_heads': 16,
        'mlp_ratio': 4,
        'qkv_bias': False,
        'qk_scale': None,
        'drop_rate': 0.1,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.0,
        'norm_layer': 'nn.LayerNorm',
        'encoder_cls_token': False,
        'decoder_cls_token': False,
        'sr_kwargs': {},
        'use_clip': False,
        'arch_encoder': 'vits',
        'arch_decoder': 'vitb',
        'encoder_lr': 1e-4,
        'encoder_weight_decay': 0.001,
        'dim_up_mlp_as_func': False,
        'decoder_load_pretrained': False,
        'ldm_z_channels': 4,
        'z_channels': 12,
        'return_all_dit_layers': False,
        'gs_rendering': False,
        'triplane_fg_bg': False,
        'density_reg': 0.0,
        'density_reg_p_dist': 0.004,
        'reg_type': 'l1',
        'triplane_decoder_lr': 1e-4,
        'super_resolution_lr': 1e-4,
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
        'rendering_kwargs': {
            'image_resolution': 256,
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
            'depth_resolution': 80,
            'depth_resolution_importance': 80,
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
            'grid_res': 256,
            'grid_scale': 2.005,
        },
        'bcg_synthesis': False,
        'bcg_synthesis_kwargs': {},
        'vit_decoder_lr': 5e-5,
        'vit_decoder_wd': 0.001,
        'ae_classname': 'vit.vit_triplane.ft',
        'depth': 6,
        'sr_ratio': 2,
        'ldm_embed_dim': 4,
        'dataset_size': -1,
        }

    
    # added by ywchen
    vqvae_pretrained_path: str = None
    data_dir: str = ''
    LN3DiffConfig = Dict_to_class(**LN3Diff_kwargs)
    ar_ckpt_path: str = None


    
    def compile_model(self, m, fast):
        # True here
        if fast == 0 or self.local_debug:
            return m
        return torch.compile(m, mode={
            1: 'reduce-overhead',
            2: 'max-autotune',
            3: 'default',
        }[fast]) if hasattr(torch, 'compile') else m
    
    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        for k in self.class_variables.keys():
            if k not in {'device'}:     # these are not serializable
                d[k] = getattr(self, k)
        return d
    
    def load_state_dict(self, d: Union[OrderedDict, dict, str]):
        if isinstance(d, str):  # for compatibility with old version
            d: dict = eval('\n'.join([l for l in d.splitlines() if '<bound' not in l and 'device(' not in l]))
        for k in d.keys():
            try:
                setattr(self, k, d[k])
            except Exception as e:
                print(f'k={k}, v={d[k]}')
                raise e
    
    @staticmethod
    def set_tf32(tf32: bool):
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')
                print(f'[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}')
            print(f'[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}')
            print(f'[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')
    
    def dump_log(self):
        if not dist.is_local_master():
            return
        if '1/' in self.cur_ep: # first time to dump log
            with open(self.log_txt_path, 'w') as fp:
                json.dump({'is_master': dist.is_master(), 'name': self.exp_name, 'cmd': self.cmd, 'commit': self.commit_id, 'branch': self.branch, 'tb_log_dir_path': self.tb_log_dir_path}, fp, indent=0)
                fp.write('\n')
        
        log_dict = {}
        for k, v in {
            'it': self.cur_it, 'ep': self.cur_ep,
            'lr': self.cur_lr, 'wd': self.cur_wd, 'grad_norm': self.grad_norm,
            'L_mean': self.L_mean, 'L_tail': self.L_tail, 'acc_mean': self.acc_mean, 'acc_tail': self.acc_tail,
            'vL_mean': self.vL_mean, 'vL_tail': self.vL_tail, 'vacc_mean': self.vacc_mean, 'vacc_tail': self.vacc_tail,
            'remain_time': self.remain_time, 'finish_time': self.finish_time,
        }.items():
            if hasattr(v, 'item'): v = v.item()
            log_dict[k] = v
        with open(self.log_txt_path, 'a') as fp:
            fp.write(f'{log_dict}\n')
    
    def __str__(self):
        s = []
        for k in self.class_variables.keys():
            if k not in {'device', 'dbg_ks_fp'}:     # these are not serializable
                s.append(f'  {k:20s}: {getattr(self, k)}')
        s = '\n'.join(s)
        return f'{{\n{s}\n}}\n'



def init_dist_and_get_args():
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith('--local-rank=') or sys.argv[i].startswith('--local_rank='):
            del sys.argv[i]
            break
    args = Args(explicit_bool=True).parse_args(known_only=True)
    if args.local_debug:
        args.pn = '1_2_3'
        args.seed = 1
        args.aln = 1e-2
        args.alng = 1e-5
        args.saln = False
        args.afuse = False
        args.pg = 0.8
        args.pg0 = 1
    else:
        pass
        # if args.data_path == '/path/to/imagenet':
        #     raise ValueError(f'{"*"*40}  please specify --data_path=/path/to/imagenet  {"*"*40}')
    
    # warn args.extra_args
    if len(args.extra_args) > 0:
        print(f'======================================================================================')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================\n{args.extra_args}')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================')
        print(f'======================================================================================\n\n')
    
    # init torch distributed
    from utils import misc
    os.makedirs(args.local_out_dir_path, exist_ok=True)
    misc.init_distributed_mode(local_out_path=args.local_out_dir_path, timeout=30)
    
    # set env
    args.set_tf32(args.tf32)
    args.seed_everything(benchmark=args.pg == 0)
    
    # update args: data loading
    args.device = dist.get_device()
    if args.pn == '256':
        args.pn = '1_2_3_4_5_6_8_10_13_16'
    elif args.pn == '512':
        args.pn = '1_2_3_4_6_9_13_18_24_32'
    elif args.pn == '1024':
        args.pn = '1_2_3_4_5_7_9_12_16_21_27_36_48_64'
    args.patch_nums = tuple(map(int, args.pn.replace('-', '_').split('_')))
    args.resos = tuple(pn * args.patch_size for pn in args.patch_nums)
    args.data_load_reso = max(args.resos)
    
    # update args: bs and lr
    bs_per_gpu = round(args.bs / args.ac / dist.get_world_size())
    args.batch_size = bs_per_gpu
    args.bs = args.glb_batch_size = args.batch_size * dist.get_world_size()
    args.num_workers = min(max(0, args.num_workers), args.batch_size)
    
    # args.tlr = args.ac * args.tblr * args.glb_batch_size / 256
    args.tlr = args.tblr
    args.twde = args.twde or args.twd
    
    if args.wp == 0:
        args.wp = args.ep * 1/50
    
    # update args: progressive training
    if args.pgwp == 0:
        args.pgwp = args.ep * 1/300
    if args.pg > 0:
        args.sche = f'lin{args.pg:g}'
    
    # update args: paths
    args.log_txt_path = os.path.join(args.local_out_dir_path, 'log.txt')
    args.last_ckpt_path = os.path.join(args.local_out_dir_path, f'ar-ckpt-last.pth')
    _reg_valid_name = re.compile(r'[^\w\-+,.]')
    tb_name = _reg_valid_name.sub(
        '_',
        f'tb-VARd{args.depth}'
        f'__pn{args.pn}'
        f'__b{args.bs}ep{args.ep}{args.opt[:4]}lr{args.tblr:g}wd{args.twd:g}'
    )
    args.tb_log_dir_path = os.path.join(args.local_out_dir_path, tb_name)
    
    return args
