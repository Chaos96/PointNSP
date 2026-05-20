import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import utils.dist as dist
from models import VAR
from vit.quant import VectorQuantizer2
from nsr.script_util import AE as VQVAE
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class VARTrainer(object):
    """Trainer class for Vector Autoregressive (VAR) model"""
    
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, var_wo_ddp: VAR, var: DDP,
        var_opt: AmpOptimizer, label_smooth: float,
        dino_image_processor=None, dino_image_model=None
    ):
        """Initialize VAR trainer
        
        Args:
            device: Device to run on
            patch_nums: Tuple of patch numbers for each resolution
            resos: Tuple of resolutions
            vae_local: Local VQVAE model
            var_wo_ddp: VAR model without DDP
            var: DDP wrapped VAR model
            var_opt: Optimizer with AMP support
            label_smooth: Label smoothing factor
            dino_image_processor: DINO image processor
            dino_image_model: DINO image model
        """
        super(VARTrainer, self).__init__()
        
        # Initialize models
        self.var = var
        self.vae_local = vae_local
        self.quantize_local = vae_local.decoder.superresolution.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp: VAR = var_wo_ddp
        self.var_opt = var_opt
        
        # Delete RNG from var_wo_ddp and create new one
        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)
        
        # Setup loss functions
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        
        # Calculate total number of patches
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        
        # Initialize loss weights
        self.loss_weight = torch.ones(1, self.L * 3, device=device) / (self.L * 3)
        
        # Store patch numbers and resolutions
        self.patch_nums = patch_nums
        self.resos = resos
        
        # Calculate begin/end indices for each resolution
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn * 3))
            cur += pn * pn * 3
        
        # Progressive training variables
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True

        # DINO models
        self.dino_image_processor = dino_image_processor
        self.dino_image_model = dino_image_model
        
    @torch.no_grad()
    def eval_ep_3D_VAR(self, ld_val: DataLoader):
        """Evaluate 3D VAR model for one epoch
        
        Args:
            ld_val: Validation dataloader
            
        Returns:
            Tuple of (triplane, caption, data)
        """
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        
        cfg = 4
        more_smooth = False
        
        for data in ld_val:
            B, V = data["image_dino_embedding"].shape[0], self.vae_local.decoder.superresolution.quantize.vocab_size
            
            # Get DINO embeddings
            pooler_output = data["image_dino_pooler_output"].to(dist.get_device(), non_blocking=True)
            dino_embeddings = data["image_dino_embedding"].to(dist.get_device(), non_blocking=True)
            caption = None
            
            with torch.inference_mode():
                triplane, _ = self.var_wo_ddp.autoregressive_infer_cfg_3D_VAR_image_l2norm(
                    B=B, dino_image_embeddings=dino_embeddings, pooler_output=pooler_output, 
                    cfg=cfg, top_k=900, top_p=0.95, more_smooth=more_smooth
                )
            break
            
        self.var_wo_ddp.train(training)
        return triplane, caption, data

    @torch.no_grad()
    def eval_ep_3D_VAR_text(self, ld_val: DataLoader):
        """Evaluate 3D VAR model with text conditioning for one epoch.
        
        Args:
            ld_val: Validation dataloader
            
        Returns:
            Tuple of (triplane, caption, data)
        """
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        
        cfg = 4  # Classifier-free guidance scale
        more_smooth = False
        
        for data in ld_val:
            # Get batch size and vocab size
            B = data["text_embedding"].shape[0]
            V = self.vae_local.decoder.superresolution.quantize.vocab_size
            
            # Get text embeddings
            pooler_output = data["text_pooler_output"].to(dist.get_device(), non_blocking=True)
            text_embeddings = data["text_embedding"].to(dist.get_device(), non_blocking=True)
            caption = data["caption"]
            
            # Generate triplane with classifier-free guidance
            with torch.inference_mode():
                triplane, _ = self.var_wo_ddp.autoregressive_infer_cfg_3D_VAR_text_l2norm(
                    B=B,
                    dino_image_embeddings=text_embeddings,
                    pooler_output=pooler_output,
                    cfg=cfg,
                    top_k=900,
                    top_p=0.95,
                    more_smooth=more_smooth
                )
            break
            
        self.var_wo_ddp.train(training)
        return triplane, caption, data
    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
        empty_pooler_output, empty_dino_image_embedding
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        """Training step
        
        Args:
            it: Current iteration
            g_it: Global iteration
            stepping: Whether to step optimizer
            metric_lg: Metric logger
            tb_lg: Tensorboard logger
            inp_B3HW: Input tensor
            label_B: Labels
            prog_si: Progressive training stage index
            prog_wp_it: Progressive warmup iterations
            empty_pooler_output: Empty pooler output for classifier-free guidance
            empty_dino_image_embedding: Empty DINO embedding for classifier-free guidance
            
        Returns:
            Tuple of (gradient norm, scale log2)
        """
        # Progressive training setup
        self.var_wo_ddp.prog_si = self.vae_local.decoder.superresolution.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: 
                self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        
        # Calculate progressive warmup
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog:
            prog_wp = 1
        if prog_si == len(self.patch_nums) - 1:
            prog_si = -1
            
        # Get batch size and vocab size
        B, V = label_B.shape[0], self.vae_local.decoder.superresolution.quantize.vocab_size
        self.var.require_backward_grad_sync = stepping

        # Get ground truth and VAR input
        gt_BL = inp_B3HW["gt_BL"].to(dist.get_device(), non_blocking=True)
        x_BLCv_wo_first_l = inp_B3HW["x_BLCv_wo_first_l"].to(dist.get_device(), non_blocking=True)

        # Get DINO embeddings
        dino_image_pooler_output = inp_B3HW["image_dino_pooler_output"].to(dist.get_device(), non_blocking=True)
        dino_image_embedding = inp_B3HW["image_dino_embedding"].to(dist.get_device(), non_blocking=True)

        # Forward pass with AMP
        with self.var_opt.amp_ctx:
            self.var_wo_ddp.forward
            
            # Get predictions with classifier-free guidance
            logits_BLV = self.var(
                pooler_output=dino_image_pooler_output,
                dino_condition=dino_image_embedding, 
                x_BLCv_wo_first_l=x_BLCv_wo_first_l,
                empty_pooler_output=empty_pooler_output,
                empty_dino_embedding=empty_dino_image_embedding
            )
            
            # Calculate loss
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            
            # Apply progressive training loss weights if needed
            if prog_si >= 0:
                bg, ed = self.begin_ends[prog_si]
                assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:
                lw = self.loss_weight
                
            loss = loss.mul(lw).sum(dim=-1).mean()

        # Backward pass
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        # Logging
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            
            if prog_si >= 0:
                Ltail = acc_tail = -1
            else:
                Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item()
                acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
                
            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)

        # Tensorboard logging
        if g_it == 0 or (g_it + 1) % 500 == 0:
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            
            if dist.is_master():
                if g_it == 0:
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)
                    
                kw = dict(z_voc_usage=cluster_usage)
                for si, (bg, ed) in enumerate(self.begin_ends):
                    if 0 <= prog_si < si:
                        break
                    pred, tar = logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'acc_{self.resos[si]}'] = acc
                    kw[f'L_{self.resos[si]}'] = ce
                    
                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
                tb_lg.update(head='AR_iter_schedule', prog_a_reso=self.resos[prog_si], prog_si=prog_si, prog_wp=prog_wp, step=g_it)
        
        self.var_wo_ddp.prog_si = self.vae_local.decoder.superresolution.quantize.prog_si = -1
        return grad_norm, scale_log2


    def train_step_text(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
        empty_text_pooler_output, empty_text_embedding
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # Set progressive training state
        self.var_wo_ddp.prog_si = self.vae_local.decoder.superresolution.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1    # No warmup for first stage
        if prog_si == len(self.patch_nums) - 1: prog_si = -1    # Max progressive stage

        # Get batch size and vocab size
        B, V = label_B.shape[0], self.vae_local.decoder.superresolution.quantize.vocab_size
        self.var.require_backward_grad_sync = stepping

        # Get ground truth indices and VAR input
        gt_BL = inp_B3HW["gt_BL"].to(dist.get_device(), non_blocking=True)
        x_BLCv_wo_first_l = inp_B3HW["x_BLCv_wo_first_l"].to(dist.get_device(), non_blocking=True)

        # Get text embeddings
        text_pooler_output = inp_B3HW["text_pooler_output"].to(dist.get_device(), non_blocking=True)
        text_embedding = inp_B3HW["text_embedding"].to(dist.get_device(), non_blocking=True)

        # Forward pass
        with self.var_opt.amp_ctx:
            self.var_wo_ddp.forward
            logits_BLV = self.var(pooler_output=text_pooler_output, dino_condition=text_embedding, 
                                x_BLCv_wo_first_l=x_BLCv_wo_first_l, empty_pooler_output=empty_text_pooler_output, 
                                empty_dino_embedding=empty_text_embedding)

            # Calculate loss
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1) 
            if prog_si >= 0:    # Progressive training
                bg, ed = self.begin_ends[prog_si]
                assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:               # Normal training
                lw = self.loss_weight
            loss = loss.mul(lw).sum(dim=-1).mean()
        
        # Backward pass
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        # Logging
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            if prog_si >= 0:    
                Ltail = acc_tail = -1
            else:               # not in progressive training
                Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item()
                acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)

        # Tensorboard logging
        if g_it == 0 or (g_it + 1) % 500 == 0:
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            if dist.is_master():
                if g_it == 0:
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)
                kw = dict(z_voc_usage=cluster_usage)
                for si, (bg, ed) in enumerate(self.begin_ends):
                    if 0 <= prog_si < si: break
                    pred, tar = logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'acc_{self.resos[si]}'] = acc
                    kw[f'L_{self.resos[si]}'] = ce
                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
                tb_lg.update(head='AR_iter_schedule', prog_a_reso=self.resos[prog_si], prog_si=prog_si, prog_wp=prog_wp, step=g_it)
        
        self.var_wo_ddp.prog_si = self.vae_local.decoder.superresolution.quantize.prog_si = -1
        return grad_norm, scale_log2
    


    def get_config(self):
        """Get trainer configuration"""
        return {
            'patch_nums': self.patch_nums,
            'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it': self.prog_it,
            'last_prog_si': self.last_prog_si,
            'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        """Get state dict for saving"""
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        """Load state dict
        
        Args:
            state: State dict to load
            strict: Whether to strictly enforce that the keys match
            skip_vae: Whether to skip loading VAE weights
        """
        for k in ('var_wo_ddp', 'vae_local'):
            if skip_vae and 'vae' in k:
                continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing: {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected: {unexpected}')

        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch: this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)
