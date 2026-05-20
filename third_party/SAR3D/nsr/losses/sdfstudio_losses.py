# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Collection of Loss Functions.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType
from torch.autograd import Variable
import numpy as np
from math import exp

# from nerfstudio.cameras.rays import RaySamples
# from nerfstudio.field_components.field_heads import FieldHeadNames

L1Loss = nn.L1Loss
MSELoss = nn.MSELoss

LOSSES = {"L1": L1Loss, "MSE": MSELoss}

EPS = 1.0e-7


def outer(
    t0_starts: TensorType[..., "num_samples_0"],
    t0_ends: TensorType[..., "num_samples_0"],
    t1_starts: TensorType[..., "num_samples_1"],
    t1_ends: TensorType[..., "num_samples_1"],
    y1: TensorType[..., "num_samples_1"],
) -> TensorType[..., "num_samples_0"]:
    """Faster version of

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L117
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L64

    Args:
        t0_starts: Start of the interval edges for t0.
        t0_ends: End of the interval edges for t0.
        t1_starts: Start of the interval edges for t1.
        t1_ends: End of the interval edges for t1.
        y1: Weights for t1.
    """
    cy1 = torch.cat([torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1)

    idx_lo = torch.searchsorted(t1_starts.contiguous(), t0_starts.contiguous(), side="right") - 1
    idx_lo = torch.clamp(idx_lo, min=0, max=y1.shape[-1] - 1)
    idx_hi = torch.searchsorted(t1_ends.contiguous(), t0_ends.contiguous(), side="right")
    idx_hi = torch.clamp(idx_hi, min=0, max=y1.shape[-1] - 1)
    cy1_lo = torch.take_along_dim(cy1[..., :-1], idx_lo, dim=-1)
    cy1_hi = torch.take_along_dim(cy1[..., 1:], idx_hi, dim=-1)
    y0_outer = cy1_hi - cy1_lo

    return y0_outer


def lossfun_outer(
    t: TensorType[..., "num_samples+1"],
    w: TensorType[..., "num_samples"],
    t_env: TensorType[..., "num_samples+1"],
    w_env: TensorType[..., "num_samples"],
):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L136
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L80

    Args:
        t: Interval edges.
        w: Weights.
        t_env: Interval edges of the upper bound enveloping histogram.
        w_env: Weights that should upper bound the inner (t,w) histogram.
    """
    w_outer = outer(t[..., :-1], t[..., 1:], t_env[..., :-1], t_env[..., 1:], w_env)
    return torch.clip(w - w_outer, min=0) ** 2 / (w + EPS)


def ray_samples_to_sdist(ray_samples):
    """Convert ray samples to s-space."""
    starts = ray_samples.spacing_starts
    ends = ray_samples.spacing_ends
    sdist = torch.cat([starts[..., 0], ends[..., -1:, 0]], dim=-1)  # (num_rays, num_samples + 1)
    return sdist


def interlevel_loss(weights_list, ray_samples_list):
    """Calculates the proposal loss in the MipNeRF-360 paper.

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/model.py#L515
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/train_utils.py#L133
    """
    c = ray_samples_to_sdist(ray_samples_list[-1]).detach()
    w = weights_list[-1][..., 0].detach()
    loss_interlevel = 0.0
    for ray_samples, weights in zip(ray_samples_list[:-1], weights_list[:-1]):
        sdist = ray_samples_to_sdist(ray_samples)
        cp = sdist  # (num_rays, num_samples + 1)
        wp = weights[..., 0]  # (num_rays, num_samples)
        loss_interlevel += torch.mean(lossfun_outer(c, w, cp, wp))
    return loss_interlevel


## zip-NeRF losses
def blur_stepfun(x, y, r):
    """Apply a step blur function with radius r."""
    x_c = torch.cat([x - r, x + r], dim=-1)
    x_r, x_idx = torch.sort(x_c, dim=-1)
    zeros = torch.zeros_like(y[:, :1])
    y_1 = (torch.cat([y, zeros], dim=-1) - torch.cat([zeros, y], dim=-1)) / (2 * r)
    x_idx = x_idx[:, :-1]
    y_2 = torch.cat([y_1, -y_1], dim=-1)[
        torch.arange(x_idx.shape[0]).reshape(-1, 1).expand(x_idx.shape).to(x_idx.device), x_idx
    ]

    y_r = torch.cumsum((x_r[:, 1:] - x_r[:, :-1]) * torch.cumsum(y_2, dim=-1), dim=-1)
    y_r = torch.cat([zeros, y_r], dim=-1)
    return x_r, y_r


def interlevel_loss_zip(weights_list, ray_samples_list):
    """Calculate the proposal loss as described in the Zip-NeRF paper."""
    c = ray_samples_to_sdist(ray_samples_list[-1]).detach()
    w = weights_list[-1][..., 0].detach()

    # 1. Normalize weights
    w_normalize = w / (c[:, 1:] - c[:, :-1])

    loss_interlevel = 0.0
    for ray_samples, weights, r in zip(ray_samples_list[:-1], weights_list[:-1], [0.03, 0.003]):
        # 2. Apply step blur with different r
        x_r, y_r = blur_stepfun(c, w_normalize, r)
        y_r = torch.clip(y_r, min=0)
        assert (y_r >= 0.0).all()

        # 3. Accumulate
        y_cum = torch.cumsum((y_r[:, 1:] + y_r[:, :-1]) * 0.5 * (x_r[:, 1:] - x_r[:, :-1]), dim=-1)
        y_cum = torch.cat([torch.zeros_like(y_cum[:, :1]), y_cum], dim=-1)

        # 4. Compute loss
        sdist = ray_samples_to_sdist(ray_samples)
        cp = sdist  # (num_rays, num_samples + 1)
        wp = weights[..., 0]  # (num_rays, num_samples)

        # Resample
        inds = torch.searchsorted(x_r, cp, side="right")
        below = torch.clamp(inds - 1, 0, x_r.shape[-1] - 1)
        above = torch.clamp(inds, 0, x_r.shape[-1] - 1)
        cdf_g0 = torch.gather(x_r, -1, below)
        bins_g0 = torch.gather(y_cum, -1, below)
        cdf_g1 = torch.gather(x_r, -1, above)
        bins_g1 = torch.gather(y_cum, -1, above)

        t = torch.clip(torch.nan_to_num((cp - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
        bins = bins_g0 + t * (bins_g1 - bins_g0)

        w_gt = bins[:, 1:] - bins[:, :-1]

        # Note: This might be unstable when wp is very small
        loss_interlevel += torch.mean(torch.clip(w_gt - wp, min=0) ** 2 / (wp + 1e-5))

    return loss_interlevel


# Verified
def lossfun_distortion(t, w):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L142
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L266
    """
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    return loss_inter + loss_intra


def distortion_loss(weights_list, ray_samples_list):
    """Compute distortion loss from mipnerf360."""
    c = ray_samples_to_sdist(ray_samples_list[-1])
    w = weights_list[-1][..., 0]
    loss = torch.mean(lossfun_distortion(c, w))
    return loss


def orientation_loss(
    weights: TensorType["bs":..., "num_samples", 1],
    normals: TensorType["bs":..., "num_samples", 3],
    viewdirs: TensorType["bs":..., 3],
):
    """Orientation loss proposed in Ref-NeRF.
    Loss that encourages that all visible normals are facing towards the camera.
    """
    w = weights
    n = normals
    v = viewdirs
    n_dot_v = (n * v[..., None, :]).sum(axis=-1)
    return (w[..., 0] * torch.fmin(torch.zeros_like(n_dot_v), n_dot_v) ** 2).sum(dim=-1)


def pred_normal_loss(
    weights: TensorType["bs":..., "num_samples", 1],
    normals: TensorType["bs":..., "num_samples", 3],
    pred_normals: TensorType["bs":..., "num_samples", 3],
):
    """Compute loss between predicted normals and normals from density."""
    return (weights[..., 0] * (1.0 - torch.sum(normals * pred_normals, dim=-1))).sum(dim=-1)


def monosdf_normal_loss(normal_pred: torch.Tensor, normal_gt: torch.Tensor):
    """Compute normal consistency loss as in monosdf.

    Args:
        normal_pred (torch.Tensor): Volume rendered normal.
        normal_gt (torch.Tensor): Monocular normal.
    """
    normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
    normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
    l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1).mean()
    cos = (1.0 - torch.sum(normal_pred * normal_gt, dim=-1)).mean()
    return l1 + cos


# Adapted from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    """Compute scale and shift for predictions."""
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    """Compute average of all valid pixels in the batch."""
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    """Compute mean of average of valid pixels in an image."""
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):
    """Compute mean squared error loss."""
    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):
    """Compute gradient loss."""
    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MiDaSMSELoss(nn.Module):
    def __init__(self, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                reduction=self.__reduction,
            )

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction="batch-based"):
        super().__init__()

        self.__data_loss = MiDaSMSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


# end copy


# copy from https://github.com/svip-lab/Indoor-SfMLearner/blob/0d682b7ce292484e5e3e2161fc9fc07e2f5ca8d1/layers.py#L218
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images."""

    def __init__(self, patch_size):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(patch_size, 1)
        self.mu_y_pool = nn.AvgPool2d(patch_size, 1)
        self.sig_x_pool = nn.AvgPool2d(patch_size, 1)
        self.sig_y_pool = nn.AvgPool2d(patch_size, 1)
        self.sig_xy_pool = nn.AvgPool2d(patch_size, 1)

        self.refl = nn.ReflectionPad2d(patch_size // 2)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class NCC(nn.Module):
    """Layer to compute the normalized cross-correlation (NCC) of patches."""

    def __init__(self, patch_size: int = 11, min_patch_variance: float = 0.01):
        super(NCC, self).__init__()
        self.patch_size = patch_size
        self.min_patch_variance = min_patch_variance

    def forward(self, x, y):
        # Convert to grayscale
        x = torch.mean(x, dim=1)
        y = torch.mean(y, dim=1)

        x_mean = torch.mean(x, dim=(1, 2), keepdim=True)
        y_mean = torch.mean(y, dim=(1, 2), keepdim=True)

        x_normalized = x - x_mean
        y_normalized = y - y_mean

        norm = torch.sum(x_normalized * y_normalized, dim=(1, 2))
        var = torch.square(x_normalized).sum(dim=(1, 2)) * torch.square(y_normalized).sum(dim=(1, 2))
        denom = torch.sqrt(var + 1e-6)

        ncc = norm / (denom + 1e-6)

        # Ignore patches with low variances
        not_valid = (torch.square(x_normalized).sum(dim=(1, 2)) < self.min_patch_variance) | (
            torch.square(y_normalized).sum(dim=(1, 2)) < self.min_patch_variance
        )
        ncc[not_valid] = 1.0

        score = 1 - ncc.clip(-1.0, 1.0)  # 0->2: smaller, better
        return score[:, None, None, None]


class MultiViewLoss(nn.Module):
    """Compute multi-view consistency loss."""

    def __init__(self, patch_size: int = 11, topk: int = 4, min_patch_variance: float = 0.01):
        super(MultiViewLoss, self).__init__()
        self.patch_size = patch_size
        self.topk = topk
        self.min_patch_variance = min_patch_variance

        self.ssim = NCC(patch_size=patch_size, min_patch_variance=min_patch_variance)

        self.iter = 0

    def forward(self, patches: torch.Tensor, valid: torch.Tensor):
        """Compute multi-view consistency loss.

        Args:
            patches (torch.Tensor): Input patches.
            valid (torch.Tensor): Validity mask for patches.

        Returns:
            torch.Tensor: Computed loss.
        """
        num_imgs, num_rays, _, num_channels = patches.shape

        if num_rays <= 0:
            return torch.tensor(0.0).to(patches.device)

        ref_patches = (
            patches[:1, ...]
            .reshape(1, num_rays, self.patch_size, self.patch_size, num_channels)
            .expand(num_imgs - 1, num_rays, self.patch_size, self.patch_size, num_channels)
            .reshape(-1, self.patch_size, self.patch_size, num_channels)
            .permute(0, 3, 1, 2)
        )  # [N_src*N_rays, 3, patch_size, patch_size]
        src_patches = (
            patches[1:, ...]
            .reshape(num_imgs - 1, num_rays, self.patch_size, self.patch_size, num_channels)
            .reshape(-1, self.patch_size, self.patch_size, num_channels)
            .permute(0, 3, 1, 2)
        )  # [N_src*N_rays, 3, patch_size, patch_size]

        # Apply same reshape to the valid mask
        src_patches_valid = (
            valid[1:, ...]
            .reshape(num_imgs - 1, num_rays, self.patch_size, self.patch_size, 1)
            .reshape(-1, self.patch_size, self.patch_size, 1)
            .permute(0, 3, 1, 2)
        )  # [N_src*N_rays, 1, patch_size, patch_size]

        ssim = self.ssim(ref_patches.detach(), src_patches)
        ssim = torch.mean(ssim, dim=(1, 2, 3))
        ssim = ssim.reshape(num_imgs - 1, num_rays)

        # Ignore invalid patches by setting SSIM error to a very large value
        ssim_valid = (
            src_patches_valid.reshape(-1, self.patch_size * self.patch_size).all(dim=-1).reshape(num_imgs - 1, num_rays)
        )
        # We should mask the error after we select the topk value, otherwise we might select far away patches that happen to be inside the image
        # ssim[torch.logical_not(ssim_valid)] = 1.1  # max ssim_error is 1

        min_ssim, idx = torch.topk(ssim, k=self.topk, largest=False, dim=0, sorted=True)

        min_ssim_valid = ssim_valid[idx, torch.arange(num_rays)[None].expand_as(idx)]

        min_ssim[torch.logical_not(min_ssim_valid)] = 0.0  # max ssim_error is 1

        return torch.sum(min_ssim) / (min_ssim_valid.float().sum() + 1e-6)



class S3IM(torch.nn.Module):
    def __init__(self, s3im_kernel_size=4, s3im_stride=4, s3im_repeat_time=10, s3im_patch_height=64, size_average=True):
        super(S3IM, self).__init__()
        self.s3im_kernel_size = s3im_kernel_size
        self.s3im_stride = s3im_stride
        self.s3im_repeat_time = s3im_repeat_time
        self.s3im_patch_height = s3im_patch_height
        self.size_average = size_average
        self.channel = 1
        self.s3im_kernel = self.create_kernel(s3im_kernel_size, self.channel)

    def gaussian(self, s3im_kernel_size, sigma):
        gauss = torch.Tensor([exp(-(x - s3im_kernel_size//2)**2/float(2*sigma**2)) for x in range(s3im_kernel_size)])
        return gauss / gauss.sum()

    def create_kernel(self, s3im_kernel_size, channel):
        _1D_window = self.gaussian(s3im_kernel_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        s3im_kernel = Variable(_2D_window.expand(channel, 1, s3im_kernel_size, s3im_kernel_size).contiguous())
        return s3im_kernel

    def _ssim(self, img1, img2, s3im_kernel, s3im_kernel_size, channel, size_average=True, s3im_stride=None):
        mu1 = F.conv2d(img1, s3im_kernel, padding=(s3im_kernel_size-1)//2, groups=channel, stride=s3im_stride)
        mu2 = F.conv2d(img2, s3im_kernel, padding=(s3im_kernel_size-1)//2, groups=channel, stride=s3im_stride)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, s3im_kernel, padding=(s3im_kernel_size-1)//2, groups=channel, stride=s3im_stride) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, s3im_kernel, padding=(s3im_kernel_size-1)//2, groups=channel, stride=s3im_stride) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, s3im_kernel, padding=(s3im_kernel_size-1)//2, groups=channel, stride=s3im_stride) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def ssim_loss(self, img1, img2):
        """Compute SSIM loss between two images.

        Args:
            img1 (torch.Tensor): First image tensor of shape [b, c, h, w].
            img2 (torch.Tensor): Second image tensor of shape [b, c, h, w].
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.s3im_kernel.data.type() == img1.data.type():
            s3im_kernel = self.s3im_kernel
        else:
            s3im_kernel = self.create_kernel(self.s3im_kernel_size, channel)

            if img1.is_cuda:
                s3im_kernel = s3im_kernel.cuda(img1.get_device())
            s3im_kernel = s3im_kernel.type_as(img1)

            self.s3im_kernel = s3im_kernel
            self.channel = channel

        return self._ssim(img1, img2, s3im_kernel, self.s3im_kernel_size, channel, self.size_average, s3im_stride=self.s3im_stride)

    def forward(self, src_vec, tar_vec):
        """Compute S3IM loss between source and target vectors.

        Args:
            src_vec (torch.Tensor): Source vector.
            tar_vec (torch.Tensor): Target vector.
        """
        loss = 0.0
        index_list = []
        for i in range(self.s3im_repeat_time):
            if i == 0:
                tmp_index = torch.arange(len(tar_vec))
                index_list.append(tmp_index)
            else:
                ran_idx = torch.randperm(len(tar_vec))
                index_list.append(ran_idx)
        res_index = torch.cat(index_list)
        tar_all = tar_vec[res_index]
        src_all = src_vec[res_index]
        tar_patch = tar_all.permute(1, 0).reshape(1, 3, self.s3im_patch_height, -1)
        src_patch = src_all.permute(1, 0).reshape(1, 3, self.s3im_patch_height, -1)
        loss = (1 - self.ssim_loss(src_patch, tar_patch))
        return loss
