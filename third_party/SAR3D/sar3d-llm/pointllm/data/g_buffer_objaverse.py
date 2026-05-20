import os
import collections
import math
import time
import itertools
import pickle
from typing import Any
import lmdb
import cv2
cv2.setNumThreads(0) # disable multiprocess
import imageio
import numpy as np
from PIL import Image
import Imath
import OpenEXR
from pathlib import Path
import torchvision
from einops import rearrange, repeat
from functools import partial
import io
from scipy.stats import special_ortho_group
import gzip
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import lz4.frame
from .nsr.volumetric_rendering.ray_sampler import RaySampler
import point_cloud_utils as pcu

import torch.multiprocessing

import copy
from pointllm import conversation as conversation_lib
from .guided_diffusion import logger
import json
import webdataset as wds
from .shapenet import LMDBDataset, LMDBDataset_MV_Compressed, decompress_and_open_image_gzip, decompress_array
from kiui.op import safe_normalize

# from .utils.gs_utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World

point_backbone_config = {
    'point_cloud_dim': 6,
    'backbone_output_dim': 8,
    'project_output_dim': 4096,
    'point_token_len': 765,
    'mm_use_point_start_end': True,
    'projection_hidden_layer': 2,
    'use_max_pool': False,
    'projection_hidden_dim': [1024, 2048],
    'default_point_patch_token': '<point_patch>',
    'point_patch_token': 32000,
    'default_point_start_token': '<point_start>',
    'default_point_end_token': '<point_end>',
    'point_start_token': 32001,
    'point_end_token': 32002
}
IGNORE_INDEX = -100

def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)"
                )
    return dict(
        input_ids=input_ids.squeeze(0),
        labels=targets.squeeze(0),
    )

def getView2World(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = C2W
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def random_rotation_matrix():
    random_rotation_3d = special_ortho_group.rvs(3)
    rotation_matrix_4x4 = np.eye(4)
    rotation_matrix_4x4[:3, :3] = random_rotation_3d
    return rotation_matrix_4x4

def matrix_to_quaternion(M: torch.Tensor) -> torch.Tensor:
    """
    Matrix-to-quaternion conversion method. Equation taken from 
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    Args:
        M: rotation matrices, (3 x 3)
    Returns:
        q: quaternion of shape (4)
    """
    tr = 1 + M[ 0, 0] + M[ 1, 1] + M[ 2, 2]

    if tr > 0:
        r = torch.sqrt(tr) / 2.0
        x = ( M[ 2, 1] - M[ 1, 2] ) / ( 4 * r )
        y = ( M[ 0, 2] - M[ 2, 0] ) / ( 4 * r )
        z = ( M[ 1, 0] - M[ 0, 1] ) / ( 4 * r )
    elif ( M[ 0, 0] > M[ 1, 1]) and (M[ 0, 0] > M[ 2, 2]):
        S = torch.sqrt(1.0 + M[ 0, 0] - M[ 1, 1] - M[ 2, 2]) * 2 # S=4*qx 
        r = (M[ 2, 1] - M[ 1, 2]) / S
        x = 0.25 * S
        y = (M[ 0, 1] + M[ 1, 0]) / S 
        z = (M[ 0, 2] + M[ 2, 0]) / S 
    elif M[ 1, 1] > M[ 2, 2]: 
        S = torch.sqrt(1.0 + M[ 1, 1] - M[ 0, 0] - M[ 2, 2]) * 2 # S=4*qy
        r = (M[ 0, 2] - M[ 2, 0]) / S
        x = (M[ 0, 1] + M[ 1, 0]) / S
        y = 0.25 * S
        z = (M[1, 2] + M[2, 1]) / S
    else:
        S = torch.sqrt(1.0 + M[ 2, 2] - M[ 0, 0] -  M[ 1, 1]) * 2 # S=4*qz
        r = (M[ 1, 0] - M[ 0, 1]) / S
        x = (M[ 0, 2] + M[ 2, 0]) / S
        y = (M[ 1, 2] + M[ 2, 1]) / S
        z = 0.25 * S

    return torch.stack([r, x, y, z], dim=-1)

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def resize_depth_mask(depth_to_resize, resolution):
    depth_resized = cv2.resize(depth_to_resize, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
    return depth_resized, depth_resized > 0

def resize_depth_mask_Tensor(depth_to_resize, resolution):
    if depth_to_resize.shape[-1] != resolution:
        depth_resized = torch.nn.functional.interpolate(
            input=depth_to_resize.unsqueeze(1),
            size=(resolution, resolution),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)
    else:
        depth_resized = depth_to_resize

    return depth_resized, depth_resized > 0

class PostProcess:
    def __init__(
        self,
        reso,
        reso_encoder,
        imgnet_normalize,
        plucker_embedding,
        decode_encode_img_only,
        mv_input,
        split_chunk_input,
        duplicate_sample,
        append_depth,
        gs_cam_format,
        orthog_duplicate,
        frame_0_as_canonical,
        pcd_path=None,
        load_pcd=False,
        split_chunk_size=8,
    ) -> None:

        self.load_pcd = load_pcd
        if pcd_path is None:  # hard-coded
            raise ValueError("pcd_path is required")

        self.frame_0_as_canonical = frame_0_as_canonical
        self.gs_cam_format = gs_cam_format
        self.append_depth = append_depth
        self.plucker_embedding = plucker_embedding
        self.decode_encode_img_only = decode_encode_img_only
        self.duplicate_sample = duplicate_sample
        self.orthog_duplicate = orthog_duplicate

        self.zfar = 100.0
        self.znear = 0.01

        transformations = []
        if not split_chunk_input:
            transformations.append(transforms.ToTensor())

        if imgnet_normalize:
            transformations.append(
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))  # type: ignore
            )
        else:
            transformations.append(
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5)))  # type: ignore

        self.normalize = transforms.Compose(transformations)
        self.reso_encoder = reso_encoder
        self.reso = reso
        self.instance_data_length = 40
        self.mv_input = mv_input
        self.split_chunk_input = split_chunk_input
        self.chunk_size = split_chunk_size if split_chunk_input else 40
        self.V = self.chunk_size // 2
        assert split_chunk_input
        self.pair_per_instance = 1
        self.ray_sampler = RaySampler()

    def gen_rays(self, c):
        intrinsics, c2w = c[16:], c[:16].reshape(4, 4)
        self.h = self.reso_encoder
        self.w = self.reso_encoder
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32) + 0.5,
            torch.arange(self.w, dtype=torch.float32) + 0.5,
            indexing='ij'
        )
        yy = yy / self.h
        xx = xx / self.w
        cx, cy, fx, fy = intrinsics[2], intrinsics[5], intrinsics[0], intrinsics[4]
        c2w = torch.from_numpy(c2w).float()
        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)
        dirs = (c2w[None, :3, :3] @ dirs)[..., 0]
        origins = c2w[None, :3, 3].expand(self.h * self.w, -1).contiguous()
        origins = origins.view(self.h, self.w, 3)
        dirs = dirs.view(self.h, self.w, 3)
        return origins, dirs

    def _post_process_batch_sample(self, sample):
        caption, ins = sample[-2:]
        instance_samples = [
            self._post_process_sample(item[instance_idx] for item in sample[:-2])
            for instance_idx in range(sample[0].shape[0])
        ]
        return (*instance_samples, caption, ins)

    def _post_process_sample(self, data_sample):
        raw_img, depth, c, bbox = data_sample
        bbox = (bbox * (self.reso / 256)).astype(np.uint8)
        img_to_encoder = (
            cv2.resize(raw_img, (self.reso_encoder, self.reso_encoder), interpolation=cv2.INTER_LANCZOS4)
            if raw_img.shape[-2] != self.reso_encoder else raw_img
        )
        img_to_encoder = self.normalize(img_to_encoder)
        if self.plucker_embedding:
            rays_o, rays_d = self.gen_rays(c)
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1).permute(2, 0, 1)
            img_to_encoder = torch.cat([img_to_encoder, rays_plucker], 0)
        img = cv2.resize(raw_img, (self.reso, self.reso), interpolation=cv2.INTER_LANCZOS4)
        img = torch.from_numpy(img).permute(2, 0, 1) / 127.5 - 1
        depth_reso, fg_mask_reso = (depth, depth) if self.decode_encode_img_only else resize_depth_mask(depth, self.reso)
        return (img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox)

    def canonicalize_pts(self, c, pcd, for_encoder=True, canonical_idx=0):
        assert c.shape[0] == self.chunk_size
        assert for_encoder
        B = c.shape[0]
        camera_poses = c[:, :16].reshape(B, 4, 4)
        cam_radius = np.linalg.norm(c[[0, self.V]][:, :16].reshape(2, 4, 4)[:, :3, 3], axis=-1, keepdims=False)
        frame1_fixed_pos = np.repeat(np.eye(4)[None], 2, axis=0)
        frame1_fixed_pos[:, 2, -1] = -cam_radius
        transform = frame1_fixed_pos @ np.linalg.inv(camera_poses[[0, self.V]])
        transform = np.expand_dims(transform, axis=1)
        repeated_homo_pcd = np.repeat(np.concatenate([pcd, np.ones_like(pcd[..., 0:1])], -1)[None], 2, axis=0)[..., None]
        new_pcd = (transform @ repeated_homo_pcd)[..., :3, 0]
        return new_pcd

    def normalize_camera(self, c, for_encoder=True, canonical_idx=0):
        assert c.shape[0] == self.chunk_size
        B = c.shape[0]
        camera_poses = c[:, :16].reshape(B, 4, 4)
        if for_encoder:
            encoder_canonical_idx = [0, self.V]
            cam_radius = np.linalg.norm(c[encoder_canonical_idx][:, :16].reshape(2, 4, 4)[:, :3, 3], axis=-1, keepdims=False)
            frame1_fixed_pos = np.repeat(np.eye(4)[None], 2, axis=0)
            frame1_fixed_pos[:, 2, -1] = -cam_radius
            transform = frame1_fixed_pos @ np.linalg.inv(camera_poses[encoder_canonical_idx])
            new_camera_poses = np.repeat(transform, self.V, axis=0) @ camera_poses
        else:
            cam_radius = np.linalg.norm(c[canonical_idx][:16].reshape(4, 4)[:3, 3], axis=-1, keepdims=False)
            frame1_fixed_pos = np.eye(4)
            frame1_fixed_pos[2, -1] = -cam_radius
            transform = frame1_fixed_pos @ np.linalg.inv(camera_poses[canonical_idx])
            new_camera_poses = np.repeat(transform[None], self.chunk_size, axis=0) @ camera_poses
        c = np.concatenate([new_camera_poses.reshape(B, 16), c[:, 16:]], axis=-1)
        return c

    def get_plucker_ray(self, c):
        rays_plucker = [
            torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1).permute(2, 0, 1)
            for idx in range(c.shape[0])
            for rays_o, rays_d in [self.gen_rays(c[idx])]
        ]
        return torch.stack(rays_plucker, 0)

    def _post_process_sample_batch(self, data_sample):
        alpha = None
        if len(data_sample) == 4:
            raw_img, depth, c, bbox = data_sample
        else:
            raw_img, depth, c, alpha, bbox = data_sample
        depth_reso, fg_mask_reso = resize_depth_mask_Tensor(torch.from_numpy(depth), self.reso)
        alpha = fg_mask_reso if alpha is None else torch.nn.functional.interpolate(
            input=torch.from_numpy(alpha / 255.0).float().unsqueeze(1),
            size=(self.reso, self.reso),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)
        bbox = (bbox * (self.reso / 256)).astype(np.uint8) if self.reso < 256 else bbox.astype(np.uint8)
        assert raw_img.shape[-2] == self.reso_encoder
        raw_img = torch.from_numpy(raw_img).permute(0, 3, 1, 2) / 255.0
        img_to_encoder = self.normalize(raw_img)
        img = torch.nn.functional.interpolate(
            input=raw_img,
            size=(self.reso, self.reso),
            mode='bilinear',
            align_corners=False,
        ) * 2 - 1 if raw_img.shape[-1] != self.reso else raw_img * 2 - 1
        if self.frame_0_as_canonical:
            encoder_canonical_idx = [0, self.V]
            c_for_encoder = self.normalize_camera(c, for_encoder=True)
            c_for_render = self.normalize_camera(c, for_encoder=False, canonical_idx=encoder_canonical_idx[0])
            c_for_render_nv = self.normalize_camera(c, for_encoder=False, canonical_idx=encoder_canonical_idx[1])
            c_for_render = np.concatenate([c_for_render, c_for_render_nv], axis=-1)
        else:
            c_for_encoder, c_for_render = c, c
        if self.plucker_embedding:
            rays_plucker = self.get_plucker_ray(c_for_encoder)
            img_to_encoder = torch.cat([img_to_encoder, rays_plucker], 1)
        if self.append_depth:
            normalized_depth = torch.from_numpy(depth).clone().unsqueeze(1)
            img_to_encoder = torch.cat([img_to_encoder, normalized_depth], 1)
        c = torch.from_numpy(c_for_render).to(torch.float32)
        return (img_to_encoder, img, alpha, depth_reso, c, torch.from_numpy(bbox))

    def rand_sample_idx(self):
        return random.randint(0, self.instance_data_length - 1)

    def rand_pair(self):
        return (self.rand_sample_idx() for _ in range(2))

    def paired_post_process(self, sample):
        all_inp_list = []
        all_nv_list = []
        caption, ins = sample[-2:]
        for _ in range(self.pair_per_instance):
            cano_idx, nv_idx = self.rand_pair()
            cano_sample = self._post_process_sample(item[cano_idx] for item in sample[:-2])
            nv_sample = self._post_process_sample(item[nv_idx] for item in sample[:-2])
            all_inp_list.extend(cano_sample)
            all_nv_list.extend(nv_sample)
        return (*all_inp_list, *all_nv_list, caption, ins)

    def get_source_cw2wT(self, source_cameras_view_to_world):
        return matrix_to_quaternion(source_cameras_view_to_world[:3, :3].transpose(0, 1))

    def c_to_3dgs_format(self, pose):
        c2w = pose[:16].reshape(4, 4)
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        fx = pose[16]
        FovX = focal2fov(fx, 1)
        tanfovx = math.tan(FovX * 0.5)
        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0
        view_world_transform = torch.tensor(getView2World(R, T, trans, scale)).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=FovX, fovY=FovX).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        c = {
            "source_cv2wT_quat": self.get_source_cw2wT(view_world_transform),
            "cam_view": world_view_transform,
            "cam_view_proj": full_proj_transform,
            "cam_pos": camera_center,
            "tanfov": tanfovx,
            "orig_pose": torch.from_numpy(pose),
            "orig_c2w": torch.from_numpy(c2w),
            "orig_w2c": torch.from_numpy(w2c),
        }
        return c

    def paired_post_process_chunk(self, sample):
        all_inp_list = []
        all_nv_list = []
        auxiliary_sample = list(sample[-2:])
        ins = sample[-1]
        assert sample[0].shape[0] == self.chunk_size
        if self.load_pcd:
            fps_pcd = pcu.load_mesh_v(str(self.pcd_path / ins / 'fps-4096.ply'))
            auxiliary_sample += [fps_pcd]
        assert self.duplicate_sample
        if self.duplicate_sample:
            shuffle_sample = sample[:-2] if self.chunk_size in [20, 12] else [
                item[np.random.permutation(self.chunk_size)] for item in sample[:-2]
            ]
            processed_sample = self._post_process_sample_batch(shuffle_sample)
            if self.load_pcd:
                if self.frame_0_as_canonical:
                    auxiliary_sample[-1] = self.canonicalize_pts(c=shuffle_sample[2], pcd=auxiliary_sample[-1], for_encoder=True)
                else:
                    auxiliary_sample[-1] = np.repeat(auxiliary_sample[-1][None], 2, axis=0)
            assert not self.orthog_duplicate
            all_inp_list.extend(item[:self.V] for item in processed_sample)
            all_nv_list.extend(item[self.V:] for item in processed_sample)
            return (*all_inp_list, *all_nv_list, *auxiliary_sample)
        else:
            processed_sample = self._post_process_sample_batch(item[:4] for item in sample[:-2])
            all_inp_list.extend(item for item in processed_sample)
            all_nv_list.extend(item for item in processed_sample)
        return (*all_inp_list, *all_nv_list, *auxiliary_sample)

    def single_sample_create_dict_noBatch(self, sample, prefix=''):
        img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox = sample
        if self.gs_cam_format:
            B, V, _ = c.shape
            c = rearrange(c, 'B V C -> (B V) C').cpu().numpy()
            all_gs_c = [self.c_to_3dgs_format(pose) for pose in c]
            c = {
                k: rearrange(torch.stack([gs_c[k] for gs_c in all_gs_c]), '(B V) ... -> B V ...', B=B, V=V)
                if isinstance(all_gs_c[0][k], torch.Tensor) else all_gs_c[0][k]
                for k in all_gs_c[0].keys()
            }
        return {
            f'{prefix}img_to_encoder': img_to_encoder,
            f'{prefix}img': img,
            f'{prefix}depth_mask': fg_mask_reso,
            f'{prefix}depth': depth_reso,
            f'{prefix}c': c,
            f'{prefix}bbox': bbox,
        }

    def single_sample_create_dict(self, sample, prefix=''):
        img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox = sample
        if self.gs_cam_format:
            B, V, _ = c.shape
            c = rearrange(c, 'B V C -> (B V) C').cpu().numpy()
            all_gs_c = [self.c_to_3dgs_format(pose) for pose in c]
            c = {
                k: rearrange(torch.stack([gs_c[k] for gs_c in all_gs_c]), '(B V) ... -> B V ...', B=B, V=V)
                if isinstance(all_gs_c[0][k], torch.Tensor) else all_gs_c[0][k]
                for k in all_gs_c[0].keys()
            }
        return {
            f'{prefix}img_to_encoder': img_to_encoder,
            f'{prefix}img': img,
            f'{prefix}depth_mask': fg_mask_reso,
            f'{prefix}depth': depth_reso,
            f'{prefix}c': c,
            f'{prefix}bbox': bbox,
        }

    def single_instance_sample_create_dict(self, sample, prfix=''):
        assert len(sample) == 42
        inp_sample_list = [[] for _ in range(6)]
        for item in sample[:40]:
            for item_idx in range(6):
                inp_sample_list[item_idx].append(item[0][item_idx])
        inp_sample = self.single_sample_create_dict(
            (torch.stack(item_list) for item_list in inp_sample_list),
            prefix=''
        )
        return {
            **inp_sample,
            'caption': sample[-2],
            'ins': sample[-1]
        }

    def decode_gzip(self, sample_pyd, shape=(256, 256)):
        raw_img, depth_alpha = sample_pyd
        raw_img = rearrange(raw_img, 'h (b w) c -> b h w c', b=self.chunk_size)
        depth = rearrange(depth, 'h (b w) c -> b h w c', b=self.chunk_size)
        alpha_mask = rearrange(alpha_mask, 'h (b w) c -> b h w c', b=self.chunk_size) / 255.0
        d_far = d_far.reshape(self.chunk_size, 1, 1, 1)
        d_near = d_near.reshape(self.chunk_size, 1, 1, 1)
        depth = 1 / ((depth / 255) * (d_far - d_near) + d_near)
        depth = depth[..., 0]
        raw_img = np.concatenate([raw_img, alpha_mask[..., 0:1]], -1)
        return raw_img, depth, c, bbox, caption, ins

    def decode_zip(self, sample_pyd):
        shape = (self.reso_encoder, self.reso_encoder)
        if isinstance(sample_pyd, tuple):
            sample_pyd = sample_pyd[0]
        assert isinstance(sample_pyd, dict)
        raw_img = decompress_and_open_image_gzip(
            sample_pyd['raw_img'],
            is_img=True,
            decompress=True,
            decompress_fn=lz4.frame.decompress
        )
        caption = sample_pyd['caption'].decode('utf-8')
        ins = sample_pyd['ins'].decode('utf-8')

        c = decompress_array(sample_pyd['c'], (
            self.chunk_size,
            25,
        ),
                             np.float32,
                             decompress=True,
                             decompress_fn=lz4.frame.decompress)

        bbox = decompress_array(
            sample_pyd['bbox'],
            (
                self.chunk_size,
                4,
            ),
            np.float32,
            decompress=True,
            decompress_fn=lz4.frame.decompress)

        if self.decode_encode_img_only:
            depth = np.zeros(shape=(self.chunk_size, *shape))
        else:
            depth = decompress_array(sample_pyd['depth'],
                                     (self.chunk_size, *shape),
                                     np.float32,
                                     decompress=True,
                                     decompress_fn=lz4.frame.decompress)

        return raw_img, depth, c, bbox, caption, ins

    def create_dict_nobatch(self, sample):
        sample_length = 6

        cano_sample_list = [[] for _ in range(sample_length)]
        nv_sample_list = [[] for _ in range(sample_length)]

        for idx in range(0, self.pair_per_instance):
            cano_sample = sample[sample_length * idx:sample_length * (idx + 1)]
            nv_sample = sample[sample_length * self.pair_per_instance +
                               sample_length * idx:sample_length *
                               self.pair_per_instance + sample_length *
                               (idx + 1)]

            for item_idx in range(sample_length):
                if self.frame_0_as_canonical:
                    if item_idx == 4:
                        cano_sample_list[item_idx].append(
                            cano_sample[item_idx][..., :25])
                        nv_sample_list[item_idx].append(
                            nv_sample[item_idx][..., :25])

                        cano_sample_list[item_idx].append(
                            nv_sample[item_idx][..., 25:])
                        nv_sample_list[item_idx].append(
                            cano_sample[item_idx][..., 25:])

                    else:
                        cano_sample_list[item_idx].append(
                            cano_sample[item_idx])
                        nv_sample_list[item_idx].append(nv_sample[item_idx])

                        cano_sample_list[item_idx].append(nv_sample[item_idx])
                        nv_sample_list[item_idx].append(cano_sample[item_idx])

                else:
                    cano_sample_list[item_idx].append(cano_sample[item_idx])
                    nv_sample_list[item_idx].append(nv_sample[item_idx])

                    cano_sample_list[item_idx].append(nv_sample[item_idx])
                    nv_sample_list[item_idx].append(cano_sample[item_idx])

        cano_sample = self.single_sample_create_dict_noBatch(
            (torch.stack(item_list, 0) for item_list in cano_sample_list),
            prefix=''
        )

        nv_sample = self.single_sample_create_dict_noBatch(
            (torch.stack(item_list, 0) for item_list in nv_sample_list),
            prefix='nv_')

        ret_dict = {
            **cano_sample,
            **nv_sample,
        }

        if not self.load_pcd:
            ret_dict.update({'caption': sample[-2], 'ins': sample[-1]})

        else:
            ret_dict.update({
                'caption': sample[-3],
                'ins': sample[-2],
                'fps_pcd': sample[-1]
            })

        return ret_dict

    def create_dict(self, sample):
        sample_length = 6

        cano_sample_list = [[] for _ in range(sample_length)]
        nv_sample_list = [[] for _ in range(sample_length)]

        for idx in range(0, self.pair_per_instance):
            cano_sample = sample[sample_length * idx:sample_length * (idx + 1)]
            nv_sample = sample[sample_length * self.pair_per_instance +
                               sample_length * idx:sample_length *
                               self.pair_per_instance + sample_length *
                               (idx + 1)]

            for item_idx in range(sample_length):
                if self.frame_0_as_canonical:
                    if item_idx == 4:
                        cano_sample_list[item_idx].append(
                            cano_sample[item_idx][..., :25])
                        nv_sample_list[item_idx].append(
                            nv_sample[item_idx][..., :25])

                        cano_sample_list[item_idx].append(
                            nv_sample[item_idx][..., 25:])
                        nv_sample_list[item_idx].append(
                            cano_sample[item_idx][..., 25:])

                    else:
                        cano_sample_list[item_idx].append(
                            cano_sample[item_idx])
                        nv_sample_list[item_idx].append(nv_sample[item_idx])

                        cano_sample_list[item_idx].append(nv_sample[item_idx])
                        nv_sample_list[item_idx].append(cano_sample[item_idx])

                else:
                    cano_sample_list[item_idx].append(cano_sample[item_idx])
                    nv_sample_list[item_idx].append(nv_sample[item_idx])

                    cano_sample_list[item_idx].append(nv_sample[item_idx])
                    nv_sample_list[item_idx].append(cano_sample[item_idx])

        cano_sample = self.single_sample_create_dict(
            (torch.cat(item_list, 0) for item_list in cano_sample_list),
            prefix='')

        nv_sample = self.single_sample_create_dict(
            (torch.cat(item_list, 0) for item_list in nv_sample_list),
            prefix='nv_')

        ret_dict = {
            **cano_sample,
            **nv_sample,
        }

        if not self.load_pcd:
            ret_dict.update({'caption': sample[-2], 'ins': sample[-1]})

        else:
            if self.frame_0_as_canonical:
                fps_pcd = rearrange(
                    sample[-1], 'B V ... -> (V B) ...')
            else:
                fps_pcd = sample[-1].repeat(
                    2, 1,
                    1)

            ret_dict.update({
                'caption': sample[-3],
                'ins': sample[-2],
                'fps_pcd': fps_pcd
            })

        return ret_dict

    def prepare_mv_input(self, sample):
        bs = len(sample['caption'])
        chunk_size = sample['img'].shape[0] // bs

        assert self.split_chunk_input

        for k, v in sample.items():
            if isinstance(v, torch.Tensor) and k != 'fps_pcd':
                sample[k] = rearrange(v, "b f c ... -> (b f) c ...",
                                      f=self.V).contiguous()

        return sample


def load_dataset(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        num_workers=6,
        load_depth=False,
        preprocess=None,
        imgnet_normalize=True,
        dataset_size=-1,
        trainer_name='input_rec',
        use_lmdb=False,
        use_wds=False,
        use_chunk=False,
        use_lmdb_compressed=False,
        infi_sampler=True):

    if use_wds:
        return load_wds_data(file_path, reso, reso_encoder, batch_size,
                             num_workers)

    if use_lmdb:
        logger.log('using LMDB dataset')

        if use_lmdb_compressed:
            if 'nv' in trainer_name:
                dataset_cls = Objv_LMDBDataset_NV_Compressed
            else:
                dataset_cls = Objv_LMDBDataset_MV_Compressed
        else:
            if 'nv' in trainer_name:
                dataset_cls = Objv_LMDBDataset_NV_NoCompressed
            else:
                dataset_cls = Objv_LMDBDataset_MV_NoCompressed

    elif use_chunk:
        dataset_cls = ChunkObjaverseDataset
    else:
        if 'nv' in trainer_name:
            dataset_cls = NovelViewObjverseDataset
        else:
            dataset_cls = MultiViewObjverseDataset

    dataset = dataset_cls(file_path,
                          reso,
                          reso_encoder,
                          test=False,
                          preprocess=preprocess,
                          load_depth=load_depth,
                          imgnet_normalize=imgnet_normalize,
                          dataset_size=dataset_size)

    logger.log('dataset_cls: {}, dataset size: {}'.format(
        trainer_name, len(dataset)))

    if use_chunk:

        def chunk_collate_fn(sample):
            default_collate_sample = torch.utils.data.default_collate(
                sample[0])
            return default_collate_sample

        collate_fn = chunk_collate_fn
    else:
        collate_fn = None

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        drop_last=False,
                        pin_memory=True,
                        persistent_workers=num_workers > 0,
                        shuffle=use_chunk,
                        collate_fn=collate_fn)
    return loader


def chunk_collate_fn(sample):
    sample = torch.utils.data.default_collate(sample)

    if 'caption' in sample:
        bs = len(sample['caption'])
        chunk_size = sample['img'].shape[0] // bs

    def merge_internal_batch(sample, merge_b_only=False):
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                if k == 'gt_BL' or k == 'image_clip_embedding' or k == 'image_clip_pooler_output' or k == 'image_dino_embedding' \
                    or k == 'x_BLCv_wo_first_l' or k == 'text_embedding' or k == 'text_pooler_output' or k=='image_dino_pooler_output' \
                    or k == 'single_image':
                    sample[k] = v
                elif v.ndim > 1:
                    if k == 'fps_pcd' or merge_b_only:
                        sample[k] = rearrange(v,
                                            "b1 b2 ... -> (b1 b2) ...").float().contiguous()

                    else:
                        sample[k] = rearrange(v,
                                            "b1 b2 f c ... -> (b1 b2 f) c ...").float().contiguous()
                elif k == 'tanfov':
                    sample[k] = v[0].float().item()
    if 'c' in sample:
        if isinstance(sample['c'], dict):
            merge_internal_batch(sample['c'], merge_b_only=True)
            merge_internal_batch(sample['nv_c'], merge_b_only=True)

    merge_internal_batch(sample)

    return sample


def load_data_3D_VAR(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        num_workers=6,
        load_depth=False,
        preprocess=None,
        imgnet_normalize=True,
        dataset_size=-1,
        trainer_name='input_rec',
        use_lmdb=False,
        use_wds=False,
        use_chunk=False,
        use_lmdb_compressed=False,
        infi_sampler=True,
        eval=False,
        load_whole=True,
        conversation_types=None,
        tokenizer=None,
        **kwargs):

    collate_fn = None

    if use_lmdb:
        logger.log('using LMDB dataset')

        if use_lmdb_compressed:
            if 'nv' in trainer_name:
                dataset_cls = Objv_LMDBDataset_NV_Compressed
            else:
                dataset_cls = Objv_LMDBDataset_MV_Compressed
        else:
            if 'nv' in trainer_name:
                dataset_cls = Objv_LMDBDataset_NV_NoCompressed
            else:
                dataset_cls = Objv_LMDBDataset_MV_NoCompressed

    elif True:
        if eval:
            dataset_cls = ChunkObjaverseDataset_eval
            collate_fn = chunk_collate_fn
        else:            
            dataset_cls = ChunkObjaverseDataset
            collate_fn = chunk_collate_fn

    else:
        if 'nv' in trainer_name:
            dataset_cls = NovelViewObjverseDataset
        else:
            dataset_cls = MultiViewObjverseDataset

    dataset = dataset_cls(file_path,
                          reso,
                          reso_encoder,
                          test=False,
                          preprocess=preprocess,
                          load_depth=load_depth,
                          imgnet_normalize=imgnet_normalize,
                          dataset_size=dataset_size,
                          load_whole=load_whole,
                          tokenizer=tokenizer,
                          **kwargs
                          )

    print('dataset_cls: {}, dataset size: {}'.format(
        trainer_name, len(dataset)))

    if infi_sampler:
        train_sampler = DistributedSampler(dataset=dataset,
                                           shuffle=True,
                                           drop_last=True)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            sampler=train_sampler,
            collate_fn=collate_fn,
        )

        return loader
    else:
        return dataset


def load_eval_data(
    file_path="",
    reso=64,
    reso_encoder=224,
    batch_size=1,
    num_workers=1,
    load_depth=False,
    preprocess=None,
    imgnet_normalize=True,
    interval=1,
    use_lmdb=False,
    plucker_embedding=False,
    load_real=False,
    four_view_for_latent=False,
    shuffle_across_cls=False,
    load_extra_36_view=False,
    gs_cam_format=False,
    single_view_for_i23d=False,
    use_chunk=False,
    **kwargs,
):

    if use_lmdb:
        logger.log('using LMDB dataset')
        dataset_cls = Objv_LMDBDataset_MV_Compressed
        dataset = dataset_cls(file_path,
                              reso,
                              reso_encoder,
                              test=True,
                              preprocess=preprocess,
                              load_depth=load_depth,
                              imgnet_normalize=imgnet_normalize,
                              interval=interval)
    elif use_chunk:
        dataset = ChunkObjaverseDataset(
            file_path,
            reso,
            reso_encoder,
            test=False,
            preprocess=preprocess,
            load_depth=load_depth,
            imgnet_normalize=imgnet_normalize,
            plucker_embedding=plucker_embedding,
            **kwargs)

    elif load_real:
        dataset = RealDataset(file_path,
                              reso,
                              reso_encoder,
                              preprocess=preprocess,
                              load_depth=load_depth,
                              test=True,
                              imgnet_normalize=imgnet_normalize,
                              interval=interval,
                              plucker_embedding=plucker_embedding)

    else:
        dataset = MultiViewObjverseDataset(
            file_path,
            reso,
            reso_encoder,
            preprocess=preprocess,
            load_depth=load_depth,
            test=True,
            imgnet_normalize=imgnet_normalize,
            interval=interval,
            plucker_embedding=plucker_embedding,
            four_view_for_latent=four_view_for_latent,
            load_extra_36_view=load_extra_36_view,
            shuffle_across_cls=shuffle_across_cls,
            gs_cam_format=gs_cam_format,
            single_view_for_i23d=single_view_for_i23d,
            **kwargs)

    print('eval dataset size: {}'.format(len(dataset)))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )
    return loader


def load_data_for_lmdb(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        num_workers=6,
        load_depth=False,
        preprocess=None,
        imgnet_normalize=True,
        dataset_size=-1,
        trainer_name='input_rec',
        shuffle_across_cls=False,
        four_view_for_latent=False,
        wds_split=1):

    dataset_cls = MultiViewObjverseDatasetforLMDB

    dataset = dataset_cls(file_path,
                          reso,
                          reso_encoder,
                          test=False,
                          preprocess=preprocess,
                          load_depth=load_depth,
                          imgnet_normalize=imgnet_normalize,
                          dataset_size=dataset_size,
                          shuffle_across_cls=shuffle_across_cls,
                          wds_split=wds_split,
                          four_view_for_latent=four_view_for_latent)

    logger.log('dataset_cls: {}, dataset size: {}'.format(
        trainer_name, len(dataset)))
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return loader, dataset.dataset_name, len(dataset)


def load_lmdb_for_lmdb(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        num_workers=6,
        load_depth=False,
        preprocess=None,
        imgnet_normalize=True,
        dataset_size=-1,
        trainer_name='input_rec'):

    dataset_cls = Objv_LMDBDataset_MV_Compressed_for_lmdb

    dataset = dataset_cls(file_path,
                          reso,
                          reso_encoder,
                          test=False,
                          preprocess=preprocess,
                          load_depth=load_depth,
                          imgnet_normalize=imgnet_normalize,
                          dataset_size=dataset_size)

    logger.log('dataset_cls: {}, dataset size: {}'.format(
        trainer_name, len(dataset)))
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
    )
    return loader, len(dataset)


def load_memory_data(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        num_workers=1,
        preprocess=None,
        imgnet_normalize=True,
        use_chunk=True,
        **kwargs):

    collate_fn = None

    if use_chunk:
        dataset_cls = ChunkObjaverseDataset
        collate_fn = chunk_collate_fn
    else:
        dataset_cls = NovelViewObjverseDataset

    dataset = dataset_cls(file_path,
                                    reso,
                                    reso_encoder,
                                    preprocess=preprocess,
                                    load_depth=True,
                                    test=False,
                                    overfitting=True,
                                    imgnet_normalize=imgnet_normalize,
                                    overfitting_bs=batch_size,
                                    **kwargs)
    logger.log('!!!!!!! memory dataset size: {} !!!!!!'.format(len(dataset)))
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )

    all_data: dict = next(
        iter(loader)
    )

    if kwargs.get('gs_cam_format', False):
        while True:
            indices = torch.randperm(
                len(dataset))[:batch_size]

            batch_c = collections.defaultdict(dict)
            for k in ['c', 'nv_c']:
                for k_c, v_c in all_data[k].items():
                    batch_c[k][k_c] = torch.index_select(
                        v_c, dim=0, index=indices).reshape(
                            batch_size, 2, *v_c.shape[2:]).float() if isinstance(
                                v_c, torch.Tensor) else v_c

            batch_data = {}
            for k, v in all_data.items():
                if k not in ['c', 'nv_c']:
                    st()
                    batch_data[k] = torch.index_select(
                        v, dim=0, index=indices).float() if isinstance(
                            v, torch.Tensor) else v

            yield {
                **batch_data,
                **batch_c,
            }

    else:
        while True:
            start_idx = np.random.randint(0, len(dataset) - batch_size + 1)
            yield {
                k: v[start_idx:start_idx + batch_size]
                for k, v in all_data.items()
            }


def read_dnormal(normald_path, cond_pos, h=None, w=None):
    cond_cam_dis = np.linalg.norm(cond_pos, 2)

    near = 0.867
    near_distance = cond_cam_dis - near

    normald = cv2.imread(normald_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    normal, depth = normald[..., :3], normald[..., 3:]

    depth[depth < near_distance] = 0

    if h is not None:
        assert w is not None
        if depth.shape[1] != h:
            depth = cv2.resize(depth, (h, w), interpolation=cv2.INTER_NEAREST
                               )
        else:
            depth = depth[..., 0]

        if normal.shape[1] != h:
            normal = cv2.resize(normal, (h, w), interpolation=cv2.INTER_NEAREST
                               )

    else:
        depth = depth[..., 0]

    return torch.from_numpy(depth).float(), torch.from_numpy(normal).float()


def get_intri(target_im=None, h=None, w=None, normalize=False):
    if target_im is None:
        assert (h is not None and w is not None)
    else:
        h, w = target_im.shape[:2]

    fx = fy = 1422.222
    res_raw = 1024
    f_x = f_y = fx * h / res_raw
    K = np.array([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
    if normalize:
        K[:6] /= h
    return K


def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return torch.from_numpy(C2W)


def read_camera_matrix_single(json_file):
    with open(json_file, 'r', encoding='utf8') as reader:
        json_content = json.load(reader)

    camera_matrix = np.eye(4)
    camera_matrix[:3, 0] = np.array(json_content['x'])
    camera_matrix[:3, 1] = np.array(json_content['y'])
    camera_matrix[:3, 2] = np.array(json_content['z'])
    camera_matrix[:3, 3] = np.array(json_content['origin'])

    return camera_matrix


def unity2blender(normal):
    normal_clone = normal.copy()
    normal_clone[..., 0] = -normal[..., -1]
    normal_clone[..., 1] = -normal[..., 0]
    normal_clone[..., 2] = normal[..., 1]

    return normal_clone


def blender2midas(img):
    img[..., 0] = -img[..., 0]
    img[..., 1] = -img[..., 1]
    img[..., -1] = -img[..., -1]
    return img


def current_milli_time():
    return round(time.time() * 1000)


class MultiViewObjverseDataset(Dataset):

    def __init__(
            self,
            file_path,
            reso,
            reso_encoder,
            preprocess=None,
            classes=False,
            load_depth=False,
            test=False,
            scene_scale=1,
            overfitting=False,
            imgnet_normalize=True,
            dataset_size=-1,
            overfitting_bs=-1,
            interval=1,
            plucker_embedding=False,
            shuffle_across_cls=False,
            wds_split=1,
            four_view_for_latent=False,
            single_view_for_i23d=False,
            load_extra_36_view=False,
            gs_cam_format=False,
            frame_0_as_canonical=False,
            **kwargs):
        self.load_extra_36_view = load_extra_36_view
        self.gs_cam_format = gs_cam_format
        self.frame_0_as_canonical = frame_0_as_canonical
        self.four_view_for_latent = four_view_for_latent
        self.single_view_for_i23d = single_view_for_i23d
        self.file_path = file_path
        self.overfitting = overfitting
        self.scene_scale = scene_scale
        self.reso = reso
        self.reso_encoder = reso_encoder
        self.classes = False
        self.load_depth = load_depth
        self.preprocess = preprocess
        self.plucker_embedding = plucker_embedding
        self.intrinsics = get_intri(h=self.reso, w=self.reso,
                                    normalize=True).reshape(9)

        assert not self.classes, "Not support class condition now."

        dataset_name = Path(self.file_path).stem.split('_')[0]
        self.dataset_name = dataset_name

        self.zfar = 100.0
        self.znear = 0.01

        def load_single_cls_instances(file_path):
            ins_list = []
            for dict_dir in os.listdir(file_path)[:]:
                for ins_dir in os.listdir(os.path.join(file_path, dict_dir)):
                    root = '/nas/shared/V2V/yslan/logs/nips23/Reconstruction/final/objav/vae/MV/170K/infer-latents/189w/v=6-rotate/latent_dir'
                    if os.path.exists(os.path.join(root,file_path.split('/')[-1], dict_dir, ins_dir, 'latent.npy') ):
                        continue
                    ins_list.append(
                        os.path.join(file_path, dict_dir, ins_dir,
                                    'campos_512_v4'))
            return ins_list

        self.ins_list = []
        if True:
            for subset in [
                    'Furnitures',
                    'daily-used',
                    'Animals',
                    'Food',
                    'Plants',
                    'Electronics',
                    'BuildingsOutdoor',
                    'Transportations_tar',
                    'Human-Shape',
            ]:
                self.ins_list += load_single_cls_instances(
                    os.path.join(self.file_path, subset))
                current_time = int(current_milli_time())
                random.seed(3407)
                random.shuffle(self.ins_list)

        else:
            self.ins_list = load_single_cls_instances(self.file_path)
            self.ins_list = sorted(self.ins_list)

        if overfitting:
            self.ins_list = self.ins_list[:1]

        self.rgb_list = []
        self.frame0_pose_list = []
        self.pose_list = []
        self.depth_list = []
        self.data_ins_list = []
        self.instance_data_length = -1

        if self.pcd_path is None:
            raise ValueError("pcd_path is not set")
        

        with open(
                '/nas/shared/V2V/yslan/aigc3d/text_captions_cap3d.json'
        ) as f:
            self.caption_data = json.load(f)

        self.shuffle_across_cls = shuffle_across_cls

        if four_view_for_latent:
            self.wds_split_all = 1
            all_ins_size = len(self.ins_list)
            ratio_size = int(all_ins_size / self.wds_split_all) + 1

            ins_list_to_process = self.ins_list[ratio_size *
                                                (wds_split):ratio_size *
                                                (wds_split + 1)]

        else:
            self.wds_split_all = 8
            all_ins_size = len(self.ins_list)
            ratio_size = all_ins_size // self.wds_split_all + 1

            ins_list_to_process = self.ins_list[ratio_size *
                                                (wds_split - 1):ratio_size *
                                                wds_split]

        uniform_sample = True
        for ins in ins_list_to_process:

            if self.four_view_for_latent:
                cur_all_fname = [f'{idx:05d}' for idx in [25,0,9,18,27,33]]
            elif self.single_view_for_i23d:
                cur_all_fname = [f'{idx:05d}'
                                 for idx in [2]]

            else:
                cur_all_fname = [t.split('.')[0] for t in os.listdir(ins)
                                 ]

                if shuffle_across_cls:
                    if uniform_sample:
                        cur_all_fname=sorted(cur_all_fname)
                        uniform_all_fname = []

                        for idx in range(8):

                            if idx % 2 == 0:
                                chunk_all_fname = [25]
                            else:
                                chunk_all_fname = [26]

                            start_1 = np.random.randint(0,7)
                            chunk_all_fname += [start_1+uniform_idx for uniform_idx in range(0,25,9)]
                            start_2 = np.random.randint(0,7) + 27
                            chunk_all_fname += [start_2, start_2 + 6]
                            assert len(chunk_all_fname) == 6
                            uniform_all_fname += [cur_all_fname[fname] for fname in chunk_all_fname] 

                        cur_all_fname = uniform_all_fname 

                    else:
                        current_time = int(current_milli_time())
                        random.seed(current_time)
                        random.shuffle(cur_all_fname)
                else:
                    cur_all_fname = sorted(cur_all_fname)

            self.frame0_pose_list += ([
                os.path.join(ins, fname, fname + '.json')
                for fname in [cur_all_fname[0]]
            ] * len(cur_all_fname))

            self.pose_list += ([
                os.path.join(ins, fname, fname + '.json')
                for fname in cur_all_fname
            ])
            self.rgb_list += ([
                os.path.join(ins, fname, fname + '.png')
                for fname in cur_all_fname
            ])

            self.depth_list += ([
                os.path.join(ins, fname, fname + '_nd.exr')
                for fname in cur_all_fname
            ])
            self.data_ins_list += ([ins] * len(cur_all_fname))

        transformations = [
            transforms.ToTensor(),
        ]
        if imgnet_normalize:
            transformations.append(
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
            )
        else:
            transformations.append(
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5)))

        self.normalize = transforms.Compose(transformations)

    def get_source_cw2wT(self, source_cameras_view_to_world):
        return matrix_to_quaternion(
            source_cameras_view_to_world[:3, :3].transpose(0, 1))

    def c_to_3dgs_format(self, pose):

        c2w = pose[:16].reshape(4, 4)

        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3])
        T = w2c[:3, 3]
        fx = pose[16]
        FovX = focal2fov(fx, 1)
        FovY = focal2fov(fx, 1)

        tanfovx = math.tan(FovX * 0.5)
        tanfovy = math.tan(FovY * 0.5)

        assert tanfovx == tanfovy

        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0

        world_view_transform = torch.tensor(getWorld2View2(R, T, trans,
                                                           scale)).transpose(
                                                               0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear,
                                                zfar=self.zfar,
                                                fovX=FovX,
                                                fovY=FovY).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
            projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        view_world_transform = torch.tensor(getView2World(R, T, trans,
                                                          scale)).transpose(
                                                              0, 1)

        c = {}
        c["source_cv2wT_quat"] = self.get_source_cw2wT(view_world_transform)
        c.update(
            cam_view=world_view_transform,
            cam_view_proj=full_proj_transform,
            cam_pos=camera_center,
            tanfov=tanfovx,
            orig_pose=torch.from_numpy(pose),
            orig_c2w=torch.from_numpy(c2w),
            orig_w2c=torch.from_numpy(w2c),
        )

        return c

    def __len__(self):
        return len(self.rgb_list)

    def load_bbox(self, mask):
        nonzero_value = torch.nonzero(mask)
        height, width = nonzero_value.max(dim=0)[0]
        top, left = nonzero_value.min(dim=0)[0]
        bbox = torch.tensor([top, left, height, width], dtype=torch.float32)
        return bbox

    def __getitem__(self, idx):

        data = self._read_data(idx)
        return data

    def gen_rays(self, c2w):
        self.h = self.reso_encoder
        self.w = self.reso_encoder
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32) + 0.5,
            torch.arange(self.w, dtype=torch.float32) + 0.5,
            indexing='ij')

        yy = yy / self.h
        xx = xx / self.w

        cx, cy, fx, fy = self.intrinsics[2], self.intrinsics[
            5], self.intrinsics[0], self.intrinsics[4]

        c2w = torch.from_numpy(c2w).float()

        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)
        del xx, yy, zz
        dirs = (c2w[None, :3, :3] @ dirs)[..., 0]

        origins = c2w[None, :3, 3].expand(self.h * self.w, -1).contiguous()
        origins = origins.view(self.h, self.w, 3)
        dirs = dirs.view(self.h, self.w, 3)

        return origins, dirs

    def normalize_camera(self, c, c_frame0):

        B = c.shape[0]
        camera_poses = c[:, :16].reshape(B, 4, 4)
        canonical_camera_poses = c_frame0[:, :16].reshape(B, 4, 4)

        cam_radius = np.linalg.norm(
            c_frame0[:, :16].reshape(1, 4, 4)[:, :3, 3],
            axis=-1,
            keepdims=False)

        frame1_fixed_pos = np.repeat(np.eye(4)[None], 1, axis=0)
        frame1_fixed_pos[:, 2, -1] = -cam_radius

        transform = frame1_fixed_pos @ np.linalg.inv(canonical_camera_poses)

        new_camera_poses = np.repeat(
            transform, 1, axis=0
        ) @ camera_poses

        c = np.concatenate([new_camera_poses.reshape(B, 16), c[:, 16:]],
                           axis=-1)
        # st()

        return c

    def _read_data(
        self,
        idx,
    ):
        rgb_fname = self.rgb_list[idx]
        pose_fname = self.pose_list[idx]

        raw_img = imageio.imread(rgb_fname)

        alpha_mask = raw_img[..., -1:] / 255
        raw_img = alpha_mask * raw_img[..., :3] + (
            1 - alpha_mask) * np.ones_like(raw_img[..., :3]) * 255

        raw_img = raw_img.astype(
            np.uint8)

        if self.preprocess is None:
            img_to_encoder = cv2.resize(raw_img,
                                        (self.reso_encoder, self.reso_encoder),
                                        interpolation=cv2.INTER_LANCZOS4)
            img_to_encoder = img_to_encoder[
                ..., :3]
            img_to_encoder = self.normalize(img_to_encoder)
        else:
            img_to_encoder = self.preprocess(Image.open(rgb_fname))

        img = cv2.resize(raw_img, (self.reso, self.reso),
                         interpolation=cv2.INTER_LANCZOS4)

        img = torch.from_numpy(img)[..., :3].permute(
            2, 0, 1
        ) / 127.5 - 1

        c2w = read_camera_matrix_single(pose_fname)

        depth, normal = read_dnormal(self.depth_list[idx], c2w[:3, 3:], self.reso,
                             self.reso)

        try:
            bbox = self.load_bbox(depth > 0)
        except:
            print(rgb_fname, flush=True)
            with open('error_log.txt', 'a') as f:
                f.write(str(rgb_fname + '\n'))
            bbox = self.load_bbox(torch.ones_like(depth))

        c = np.concatenate([c2w.reshape(16), self.intrinsics],
                           axis=0).reshape(25).astype(
                               np.float32)

        if self.frame_0_as_canonical:
            frame0_pose_name = self.frame0_pose_list[idx]
            c2w_frame0 = read_camera_matrix_single(
                frame0_pose_name)
            c = self.normalize_camera(c[None], c2w_frame0[None])[0]
            c2w = c[:16].reshape(4, 4)

        rays_o, rays_d = self.gen_rays(c2w)
        rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d],
                                 dim=-1)

        img_to_encoder = torch.cat(
            [img_to_encoder, rays_plucker.permute(2, 0, 1)],
            0).float()

        depth, normal = read_dnormal(self.depth_list[idx], c2w[:3, 3:],
                                        self.reso_encoder,
                                        self.reso_encoder)
        normalized_depth = depth.unsqueeze(0)
        img_to_encoder = torch.cat([img_to_encoder, normalized_depth],
                                   0)

        if self.gs_cam_format:
            c = self.c_to_3dgs_format(c)
        else:
            c = torch.from_numpy(c)

        ret_dict = {
            'img_to_encoder': img_to_encoder,
            'img': img,
            'c': c,
        }

        pcd_ins = Path(self.data_ins_list[idx]).relative_to(
            Path(self.file_path).parent).parent

        ins = str(
            (Path(self.data_ins_list[idx]).relative_to(self.file_path)).parent)
        caption = self.caption_data['/'.join(ins.split('/')[1:])]

        ret_dict.update({
            'depth': depth,
            'normal': normal,
            'depth_mask': depth > 0,
            'bbox': bbox,
            'caption': caption,
            'rays_plucker': rays_plucker,
            'ins': ins,
        })

        return ret_dict
class ChunkObjaverseDataset(Dataset):
    def __init__(
            self,
            file_path,
            reso,
            reso_encoder,
            preprocess=None,
            classes=False,
            load_depth=False,
            test=False,
            scene_scale=1,
            overfitting=False,
            imgnet_normalize=True,
            dataset_size=-1,
            overfitting_bs=-1,
            interval=1,
            plucker_embedding=False,
            shuffle_across_cls=False,
            wds_split=1,
            four_view_for_latent=False,
            single_view_for_i23d=False,
            load_extra_36_view=False,
            gs_cam_format=False,
            frame_0_as_canonical=False,
            split_chunk_size=12,
            mv_input=True,
            append_depth=True,
            pcd_path=None,
            load_pcd=False,
            load_whole=True,
            tokenizer=None,
            **kwargs):

        super().__init__()
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.chunk_size = 12
        self.gs_cam_format = gs_cam_format
        self.frame_0_as_canonical = frame_0_as_canonical
        self.four_view_for_latent = four_view_for_latent
        self.overfitting = overfitting
        self.scene_scale = scene_scale
        self.reso = reso
        self.reso_encoder = reso_encoder
        self.classes = False
        self.load_depth = load_depth
        self.preprocess = preprocess
        self.plucker_embedding = plucker_embedding
        self.intrinsics = get_intri(h=self.reso, w=self.reso,
                                    normalize=True).reshape(9)
        assert not self.classes, "Not support class condition now."

        dataset_name = Path(self.file_path).stem.split('_')[0]
        self.dataset_name = dataset_name

        self.zfar = 100.0
        self.znear = 0.01

        self.chunk_list = []
        self.load_whole = load_whole

        dataset_json = []
        for cl in [
                    'Furnitures',
                    'daily-used',
                    'Animals',
                    'Food',
                    'Plants',
                    'Electronics',
                    'BuildingsOutdoor',
                    'Transportations_tar',
                    'Human-Shape',
            ]:
            with open(f'{self.file_path}/dataset.json', 'r') as f:
                cl_dataset_json = json.load(f)[cl][:-100]
            dataset_json = dataset_json + cl_dataset_json     

        if self.chunk_size == 12:
            self.img_ext = 'png'
            for v in dataset_json:
                self.chunk_list.append(v)

        self.post_process = PostProcess(
            reso,
            reso_encoder,
            imgnet_normalize=imgnet_normalize,
            plucker_embedding=plucker_embedding,
            decode_encode_img_only=False,
            mv_input=mv_input,
            split_chunk_input=12,
            duplicate_sample=True,
            append_depth=append_depth,
            gs_cam_format=gs_cam_format,
            orthog_duplicate=False,
            frame_0_as_canonical=frame_0_as_canonical,
            pcd_path=pcd_path,
            load_pcd=load_pcd,
            split_chunk_size=12,
        )

    def fetch_chunk_list(self, file_path):
        if os.path.isdir(file_path):
            chunks = [
                os.path.join(file_path, fname) for fname in os.listdir(file_path)
                if fname.isdigit()
            ]
            return sorted(chunks)
        else:
            return []

    def _pre_process_chunk(self):
        pass

    def read_chunk(self, chunk_path):
        raw_img = imageio.imread(os.path.join(chunk_path, f'raw_img.{self.img_ext}'))
        h, bw, c = raw_img.shape
        raw_img = raw_img.reshape(h, self.chunk_size, -1, c).transpose(
            (1, 0, 2, 3))

        depth_alpha = imageio.imread(
            os.path.join(chunk_path, 'depth_alpha.jpg'))
        depth_alpha = depth_alpha.reshape(h * 2, self.chunk_size,
                                          -1).transpose((1, 0, 2))

        depth, alpha = np.split(depth_alpha, 2, axis=1)

        c = np.load(os.path.join(chunk_path, 'c.npy'))

        d_near_far = np.load(os.path.join(chunk_path, 'd_near_far.npy'))
        bbox = np.load(os.path.join(chunk_path, 'bbox.npy'))

        d_near = d_near_far[0].reshape(self.chunk_size, 1, 1)
        d_far = d_near_far[1].reshape(self.chunk_size, 1, 1)
        depth = 1 / ((depth / 255) * (d_far - d_near) + d_near)

        depth[depth > 2.9] = 0.0

        with open(os.path.join(chunk_path, 'caption_3dtopia.txt'), 'r', encoding="utf-8") as f:
            caption = f.read()

        with open(os.path.join(chunk_path, 'ins.txt'), 'r', encoding="utf-8") as f:
            ins = f.read()

        return raw_img, depth, c, alpha, bbox, caption, ins

    def load_latent(self, sample, latent_path=None):
        gt_BL = torch.from_numpy(np.load(os.path.join(latent_path, "gt_BL_dim_8_l2norm_lrm_256.npy")))

        sample.update({
        'gt_BL': gt_BL,
        })
        
        return sample

    def __len__(self):
        return len(self.chunk_list)

    def __getitem__(self, index) -> Any:
        if self.load_whole:
            sample = self.read_chunk(os.path.join(self.file_path, self.chunk_list[index]))
            sample = self.post_process.paired_post_process_chunk(sample)
            sample = self.post_process.create_dict_nobatch(sample)
        else:
            sample = {}

        sample.update({
            'sample_path': os.path.join(self.file_path, self.chunk_list[index]),
        })

        sample = self.load_latent(sample, os.path.join(self.file_path, self.chunk_list[index]))

        sources = dict()
        sources['conversations'] = [{'from': 'human', 'value': '<point>\nGive a concise interpretation of the 3D data presented here.'}, {'from': 'gpt', 'value': sample['caption']}]
        sources = [sources]
        sources = self.preprocess_multimodal_point_cloud(
                copy.deepcopy([e["conversations"] for e in sources]), point_backbone_config, point_indicator='<point>')
        data_dict = preprocess_v1(
            sources,
            self.tokenizer)
        data_dict["gt_BL"] = sample["gt_BL"]

        return data_dict

    def preprocess_multimodal_point_cloud(
        self,
        sources,
        point_backbone_config: dict,
        point_indicator: str = "<point>",
    ):
        point_token_len = point_backbone_config['point_token_len']
        default_point_patch_token = point_backbone_config['default_point_patch_token']

        for source in sources:
            for sentence in source:
                replace_token = default_point_patch_token * point_token_len 
                if point_backbone_config['mm_use_point_start_end']:
                    replace_token = point_backbone_config['default_point_start_token']+ replace_token + point_backbone_config['default_point_end_token']
                sentence["value"] = sentence["value"].replace(point_indicator, replace_token)

        return sources

class ChunkObjaverseDataset_eval(Dataset):
    def __init__(
            self,
            file_path,
            reso,
            reso_encoder,
            preprocess=None,
            classes=False,
            load_depth=False,
            test=False,
            scene_scale=1,
            overfitting=False,
            imgnet_normalize=True,
            dataset_size=-1,
            overfitting_bs=-1,
            interval=1,
            plucker_embedding=False,
            shuffle_across_cls=False,
            wds_split=1,
            four_view_for_latent=False,
            single_view_for_i23d=False,
            load_extra_36_view=False,
            gs_cam_format=False,
            frame_0_as_canonical=False,
            split_chunk_size=12,
            mv_input=True,
            append_depth=True,
            pcd_path=None,
            load_pcd=False,
            load_whole=True,
            tokenizer=None,
            **kwargs):

        super().__init__()
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.chunk_size = 12
        self.gs_cam_format = gs_cam_format
        self.frame_0_as_canonical = frame_0_as_canonical
        self.four_view_for_latent = four_view_for_latent
        self.overfitting = overfitting
        self.scene_scale = scene_scale
        self.reso = reso
        self.reso_encoder = reso_encoder
        self.classes = False
        self.load_depth = load_depth
        self.preprocess = preprocess
        self.plucker_embedding = plucker_embedding
        self.intrinsics = get_intri(h=self.reso, w=self.reso,
                                    normalize=True).reshape(9)
        assert not self.classes, "Not support class condition now."

        dataset_name = Path(self.file_path).stem.split('_')[0]
        self.dataset_name = dataset_name

        self.zfar = 100.0
        self.znear = 0.01

        self.chunk_list = []
        self.load_whole = load_whole

        dataset_json = []
        for cl in [
                    'Furnitures',
                    'daily-used',
                    'Animals',
                    'Food',
                    'Plants',
                    'Electronics',
                    'BuildingsOutdoor',
                    'Transportations_tar',
                    'Human-Shape',
            ]:
            with open(f'{self.file_path}/dataset.json', 'r') as f:
                cl_dataset_json = json.load(f)[cl][-100:]
            dataset_json = dataset_json + cl_dataset_json     

        if self.chunk_size == 12:
            self.img_ext = 'png'
            for v in dataset_json:
                self.chunk_list.append(v)

        self.post_process = PostProcess(
            reso,
            reso_encoder,
            imgnet_normalize=imgnet_normalize,
            plucker_embedding=plucker_embedding,
            decode_encode_img_only=False,
            mv_input=mv_input,
            split_chunk_input=12,
            duplicate_sample=True,
            append_depth=append_depth,
            gs_cam_format=gs_cam_format,
            orthog_duplicate=False,
            frame_0_as_canonical=frame_0_as_canonical,
            pcd_path=pcd_path,
            load_pcd=load_pcd,
            split_chunk_size=12,
        )

    def fetch_chunk_list(self, file_path):
        if os.path.isdir(file_path):
            chunks = [
                os.path.join(file_path, fname) for fname in os.listdir(file_path)
                if fname.isdigit()
            ]
            return sorted(chunks)
        else:
            return []

    def _pre_process_chunk(self):
        pass

    def read_chunk(self, chunk_path):
        raw_img = imageio.imread(os.path.join(chunk_path, f'raw_img.{self.img_ext}'))
        h, bw, c = raw_img.shape
        raw_img = raw_img.reshape(h, self.chunk_size, -1, c).transpose(
            (1, 0, 2, 3))

        depth_alpha = imageio.imread(
            os.path.join(chunk_path, 'depth_alpha.jpg'))
        depth_alpha = depth_alpha.reshape(h * 2, self.chunk_size,
                                          -1).transpose((1, 0, 2))

        depth, alpha = np.split(depth_alpha, 2, axis=1)

        c = np.load(os.path.join(chunk_path, 'c.npy'))

        d_near_far = np.load(os.path.join(chunk_path, 'd_near_far.npy'))
        bbox = np.load(os.path.join(chunk_path, 'bbox.npy'))

        d_near = d_near_far[0].reshape(self.chunk_size, 1, 1)
        d_far = d_near_far[1].reshape(self.chunk_size, 1, 1)
        depth = 1 / ((depth / 255) * (d_far - d_near) + d_near)

        depth[depth > 2.9] = 0.0

        with open(os.path.join(chunk_path, 'caption_3dtopia.txt'), 'r', encoding="utf-8") as f:
            caption = f.read()

        with open(os.path.join(chunk_path, 'ins.txt'), 'r', encoding="utf-8") as f:
            ins = f.read()

        return raw_img, depth, c, alpha, bbox, caption, ins

    def load_latent(self, sample, latent_path=None):
        if os.path.exists(os.path.join(latent_path, "gt_BL_dim_8_l2norm_lrm.npy")):
            gt_BL = torch.from_numpy(np.load(os.path.join(latent_path, "gt_BL_dim_8_l2norm_lrm_256.npy")))
            x_BLCv_wo_first_l = torch.from_numpy(np.load(os.path.join(latent_path, "x_BLCv_wo_first_l_dim_8_l2_norm_lrm_256.npy")))
            image_dino_embedding = torch.from_numpy(np.load(os.path.join(latent_path, "image_dino_embedding_lrm.npy")))[1:, :]
            image_dino_pooler_output = torch.from_numpy(np.load(os.path.join(latent_path, "image_dino_pooler_output_lrm.npy")))
            single_image = torch.from_numpy(np.array(Image.open(os.path.join(latent_path, "single_image.png"))))

            sample.update({
            'gt_BL': gt_BL,
            'x_BLCv_wo_first_l': x_BLCv_wo_first_l,
            'image_dino_pooler_output': image_dino_pooler_output,
            'image_dino_embedding': image_dino_embedding,
            'single_image': single_image
            })
        else:
            raise NotImplementedError(os.path.join(latent_path, "gt_BL_dim_8_l2norm_lrm_256.npy"))
        
        return sample

    def __len__(self):
        return len(self.chunk_list)

    def __getitem__(self, index) -> Any:
        if self.load_whole:
            sample = self.read_chunk(os.path.join(self.file_path, self.chunk_list[index]))
            sample = self.post_process.paired_post_process_chunk(sample)
            sample = self.post_process.create_dict_nobatch(sample)
        else:
            sample = {}

        sample.update({
            'sample_path': os.path.join(self.file_path, self.chunk_list[index]),
        })

        sample = self.load_latent(sample, os.path.join(self.file_path, self.chunk_list[index]))

        sources = dict()
        sources['conversations'] = [{'from': 'human', 'value': '<point>\nGive a concise interpretation of the 3D data presented here.'}, {'from': 'gpt', 'value': ''}]
        sources = [sources]
        sources = self.preprocess_multimodal_point_cloud(
                copy.deepcopy([e["conversations"] for e in sources]), point_backbone_config, point_indicator='<point>')
        data_dict = self.preprocess_v1(
            sources,
            self.tokenizer)
        data_dict["gt_BL"] = sample["gt_BL"]
        data_dict["sample_path"] = sample["sample_path"]
        data_dict["gt_caption"] = sample["caption"]

        return data_dict

    def preprocess_v1(
        self,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        ):
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()
        assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

        return dict(
            input_ids=input_ids.squeeze(0),
            labels=targets.squeeze(0),
        )

    def preprocess_multimodal_point_cloud(
        self,
        sources,
        point_backbone_config: dict,
        point_indicator: str = "<point>",
    ):
        point_token_len = point_backbone_config['point_token_len']
        default_point_patch_token = point_backbone_config['default_point_patch_token']

        for source in sources:
            for sentence in source:
                replace_token = default_point_patch_token * point_token_len 
                if point_backbone_config['mm_use_point_start_end']:
                    replace_token = point_backbone_config['default_point_start_token']+ replace_token + point_backbone_config['default_point_end_token']
                sentence["value"] = sentence["value"].replace(point_indicator, replace_token)

        return sources

class RealDataset(Dataset):

    def __init__(
            self,
            file_path,
            reso,
            reso_encoder,
            preprocess=None,
            classes=False,
            load_depth=False,
            test=False,
            scene_scale=1,
            overfitting=False,
            imgnet_normalize=True,
            dataset_size=-1,
            overfitting_bs=-1,
            interval=1,
            plucker_embedding=False,
            shuffle_across_cls=False,
            wds_split=1,
    ) -> None:
        super().__init__()

        self.file_path = file_path
        self.overfitting = overfitting
        self.scene_scale = scene_scale
        self.reso = reso
        self.reso_encoder = reso_encoder
        self.classes = False
        self.load_depth = load_depth
        self.preprocess = preprocess
        self.plucker_embedding = plucker_embedding

        self.rgb_list = []

        all_fname = [
            t for t in os.listdir(self.file_path)
            if t.split('.')[1] in ['png', 'jpg']
        ]
        self.rgb_list += ([
            os.path.join(self.file_path, fname) for fname in all_fname
        ])

        transformations = [
            transforms.ToTensor(),
        ]

        assert imgnet_normalize
        if imgnet_normalize:
            transformations.append(
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
            )
        else:
            transformations.append(
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5)))

        self.normalize = transforms.Compose(transformations)
        camera = torch.load('eval_pose.pt', map_location='cpu')
        self.eval_camera = camera

        self.calc_rays_plucker()

    def gen_rays(self, c):
        intrinsics, c2w = c[16:], c[:16].reshape(4, 4)
        self.h = self.reso_encoder
        self.w = self.reso_encoder
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32) + 0.5,
            torch.arange(self.w, dtype=torch.float32) + 0.5,
            indexing='ij')

        yy = yy / self.h
        xx = xx / self.w

        cx, cy, fx, fy = intrinsics[2], intrinsics[5], intrinsics[
            0], intrinsics[4]

        if not isinstance(c2w, torch.Tensor):
            c2w = torch.from_numpy(c2w)

        c2w = c2w.float()

        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)
        del xx, yy, zz
        dirs = (c2w[None, :3, :3] @ dirs)[..., 0]

        origins = c2w[None, :3, 3].expand(self.h * self.w, -1).contiguous()
        origins = origins.view(self.h, self.w, 3)
        dirs = dirs.view(self.h, self.w, 3)

        return origins, dirs

    def calc_rays_plucker(self):
        all_rays_plucker = []

        for c2w in self.eval_camera:
            rays_o, rays_d = self.gen_rays(c2w)
            rays_plucker = torch.cat(
                [torch.cross(rays_o, rays_d, dim=-1), rays_d],
                dim=-1)
            all_rays_plucker.append(rays_plucker)

        self.all_rays_plucker = torch.stack(all_rays_plucker,
                                            0).permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, index) -> Any:
        rgb_fname = self.rgb_list[index]

        raw_img = imageio.imread(rgb_fname)

        if raw_img.shape[-1] == 4:
            alpha_mask = raw_img[..., 3:4] / 255.0
            bg_white = np.ones_like(alpha_mask) * 255.0
            raw_img = raw_img[..., :3] * alpha_mask + (
                1 - alpha_mask) * bg_white
            raw_img = raw_img.astype(np.uint8)

        img_to_encoder = cv2.resize(raw_img,
                                    (self.reso_encoder, self.reso_encoder),
                                    interpolation=cv2.INTER_LANCZOS4)

        img_to_encoder = self.normalize(img_to_encoder)

        img_to_encoder = torch.cat(
            [img_to_encoder, self.all_rays_plucker[index]],
            0)

        img = cv2.resize(raw_img, (self.reso, self.reso),
                         interpolation=cv2.INTER_LANCZOS4)

        img = torch.from_numpy(img)[..., :3].permute(
            2, 0, 1
        ) / 127.5 - 1

        ret_dict = {
            'img_to_encoder':
            img_to_encoder.unsqueeze(0).repeat_interleave(40, 0),
            'img': img.unsqueeze(0).repeat_interleave(40, 0),
            'c': self.eval_camera,
            'ins': 'placeholder',
            'bbox': 'placeholder',
            'caption': 'placeholder',
        }

        return ret_dict


class NovelViewObjverseDataset(MultiViewObjverseDataset):

    def __init__(self,
                 file_path,
                 reso,
                 reso_encoder,
                 preprocess=None,
                 classes=False,
                 load_depth=False,
                 test=False,
                 scene_scale=1,
                 overfitting=False,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 overfitting_bs=-1,
                 **kwargs):
        super().__init__(file_path, reso, reso_encoder, preprocess, classes,
                         load_depth, test, scene_scale, overfitting,
                         imgnet_normalize, dataset_size, overfitting_bs,
                         **kwargs)

    def __getitem__(self, idx):
        input_view = super().__getitem__(idx)

        novel_view = super().__getitem__(
            (idx // self.instance_data_length) * self.instance_data_length +
            random.randint(0, self.instance_data_length - 1))

        input_view.update({f'nv_{k}': v for k, v in novel_view.items()})
        return input_view


class MultiViewObjverseDatasetforLMDB(MultiViewObjverseDataset):

    def __init__(
        self,
        file_path,
        reso,
        reso_encoder,
        preprocess=None,
        classes=False,
        load_depth=False,
        test=False,
        scene_scale=1,
        overfitting=False,
        imgnet_normalize=True,
        dataset_size=-1,
        overfitting_bs=-1,
        shuffle_across_cls=False,
        wds_split=1,
        four_view_for_latent=False,
    ):
        super().__init__(file_path,
                         reso,
                         reso_encoder,
                         preprocess,
                         classes,
                         load_depth,
                         test,
                         scene_scale,
                         overfitting,
                         imgnet_normalize,
                         dataset_size,
                         overfitting_bs,
                         shuffle_across_cls=shuffle_across_cls,
                         wds_split=wds_split,
                         four_view_for_latent=four_view_for_latent)

        assert self.reso == 256

        with open(
                '/nas/shared/V2V/yslan/aigc3d/text_captions_cap3d.json'
        ) as f:
            self.caption_data = json.load(f)

    def __len__(self):
        return super().__len__()

    def quantize_depth(self, depth):
        bg = depth == 0
        depth[bg] = 3
        disparity = 1 / depth

        far = disparity.max().item()
        near = disparity.min().item()

        d_normalized = (disparity - near) / (far - near)
        d_normalized = np.nan_to_num(d_normalized.cpu().numpy())
        d_normalized = (np.clip(d_normalized, 0, 1) * 255).astype(
            np.uint8)

        return d_normalized, near, far

    def __getitem__(self, idx):
        rgb_fname = self.rgb_list[idx]
        pose_fname = self.pose_list[idx]
        raw_img = imageio.imread(rgb_fname)

        assert raw_img.shape[-1] == 4

        alpha_mask = raw_img[..., -1:] / 255

        raw_img = alpha_mask * raw_img[..., :3] + (
            1 - alpha_mask) * np.ones_like(raw_img[..., :3]) * 255

        raw_img = np.concatenate([raw_img, alpha_mask * 255], -1)
        raw_img = raw_img.astype(np.uint8)

        raw_img = cv2.resize(raw_img, (self.reso, self.reso),
                             interpolation=cv2.INTER_LANCZOS4)
        alpha_mask = raw_img[..., -1] / 255
        raw_img = raw_img[..., :3]

        c2w = read_camera_matrix_single(pose_fname)
        c = np.concatenate([c2w.reshape(16), self.intrinsics],
                           axis=0).reshape(25).astype(
                               np.float32)
        c = torch.from_numpy(c)

        depth, normal = read_dnormal(self.depth_list[idx], c2w[:3, 3:], self.reso,
                             self.reso)

        d_normalized, d_near, d_far = self.quantize_depth(depth)

        bbox = self.load_bbox(torch.from_numpy(alpha_mask) > 0)

        ins = str(
            (Path(self.data_ins_list[idx]).relative_to(self.file_path)).parent)
        caption = self.caption_data['/'.join(ins.split('/')[1:])]

        ret_dict = {
            'normal': normal,
            'raw_img': raw_img,
            'c': c,
            'bbox': bbox,
            'ins': ins,
            'caption': caption,
            'alpha_mask': alpha_mask,
            'd_normalized': d_normalized,
            'd_near': d_near,
            'd_far': d_far,
        }
        return ret_dict


class Objv_LMDBDataset_MV_Compressed(LMDBDataset_MV_Compressed):

    def __init__(self,
                 lmdb_path,
                 reso,
                 reso_encoder,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 test=False,
                 **kwargs):
        super().__init__(lmdb_path,
                         reso,
                         reso_encoder,
                         imgnet_normalize,
                         dataset_size=dataset_size,
                         **kwargs)
        self.instance_data_length = 40
        if test:
            self.length = self.instance_data_length
        elif dataset_size > 0:
            self.length = dataset_size * self.instance_data_length

        with open(
                '/cpfs01/shared/V2V/V2V_hdd/yslan/aigc3d/text_captions_cap3d.json'
        ) as f:
            self.caption_data = json.load(f)
        with open(os.path.join(lmdb_path, 'idx_to_ins_mapping.json')) as f:
            self.idx_to_ins_mapping = json.load(f)

    def _load_data(self, idx):
        raw_img, depth, c, bbox = self._load_lmdb_data(idx)

        caption = self.caption_data[self.idx_to_ins_mapping[str(idx)]]

        return {
            **self._post_process_sample(raw_img, depth),
            'c': c,
            'bbox': (bbox * (self.reso / 512.0)).astype(np.uint8),
            'caption': caption
        }

    def __getitem__(self, idx):
        return self._load_data(idx)


class Objv_LMDBDataset_MV_NoCompressed(Objv_LMDBDataset_MV_Compressed):

    def __init__(self,
                 lmdb_path,
                 reso,
                 reso_encoder,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 test=False,
                 **kwargs):
        super().__init__(lmdb_path, reso, reso_encoder, imgnet_normalize,
                         dataset_size, test, **kwargs)

    def _load_data(self, idx):
        raw_img, depth, c, bbox = self._load_lmdb_data_no_decompress(idx)

        caption = self.caption_data[self.idx_to_ins_mapping[str(idx)]]

        return {
            **self._post_process_sample(raw_img, depth), 'c': c,
            'bbox': (bbox * (self.reso / 512.0)).astype(np.uint8),
            'caption': caption
        }
        return {}


class Objv_LMDBDataset_NV_NoCompressed(Objv_LMDBDataset_MV_NoCompressed):

    def __init__(self,
                 lmdb_path,
                 reso,
                 reso_encoder,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 test=False,
                 **kwargs):
        super().__init__(lmdb_path, reso, reso_encoder, imgnet_normalize,
                         dataset_size, test, **kwargs)

    def __getitem__(self, idx):
        input_view = self._load_data(idx)

        try:
            novel_view = self._load_data(
                (idx // self.instance_data_length) *
                self.instance_data_length +
                random.randint(0, self.instance_data_length - 1))
        except Exception as e:
            raise NotImplementedError(idx)

        input_view.update({f'nv_{k}': v for k, v in novel_view.items()})
        return input_view


class Objv_LMDBDataset_MV_Compressed_for_lmdb(LMDBDataset_MV_Compressed):

    def __init__(self,
                 lmdb_path,
                 reso,
                 reso_encoder,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 test=False,
                 **kwargs):
        super().__init__(lmdb_path,
                         reso,
                         reso_encoder,
                         imgnet_normalize,
                         dataset_size=dataset_size,
                         **kwargs)
        self.instance_data_length = 40
        if test:
            self.length = self.instance_data_length
        elif dataset_size > 0:
            self.length = dataset_size * self.instance_data_length

        with open(
                '/cpfs01/shared/V2V/V2V_hdd/yslan/aigc3d/text_captions_cap3d.json'
        ) as f:
            self.caption_data = json.load(f)
        with open(os.path.join(lmdb_path, 'idx_to_ins_mapping.json')) as f:
            self.idx_to_ins_mapping = json.load(f)

    def load_bbox(self, mask):
        nonzero_value = torch.nonzero(mask)
        height, width = nonzero_value.max(dim=0)[0]
        top, left = nonzero_value.min(dim=0)[0]
        bbox = torch.tensor([top, left, height, width], dtype=torch.float32)
        return bbox

    def __getitem__(self, idx):
        raw_img, depth, c, bbox = self._load_lmdb_data(idx)
        return {'raw_img': raw_img, 'depth': depth, 'c': c, 'bbox': bbox}


class Objv_LMDBDataset_NV_Compressed(Objv_LMDBDataset_MV_Compressed):

    def __init__(self,
                 lmdb_path,
                 reso,
                 reso_encoder,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 **kwargs):
        super().__init__(lmdb_path, reso, reso_encoder, imgnet_normalize,
                         dataset_size, **kwargs)

    def __getitem__(self, idx):
        input_view = self._load_data(idx)

        try:
            novel_view = self._load_data(
                (idx // self.instance_data_length) *
                self.instance_data_length +
                random.randint(0, self.instance_data_length - 1))
        except Exception as e:
            raise NotImplementedError(idx)

        input_view.update({f'nv_{k}': v for k, v in novel_view.items()})
        return input_view


def load_wds_ResampledShard(file_path,
                            batch_size,
                            num_workers,
                            reso,
                            reso_encoder,
                            test=False,
                            preprocess=None,
                            imgnet_normalize=True,
                            plucker_embedding=False,
                            decode_encode_img_only=False,
                            load_instance=False,
                            mv_input=False,
                            split_chunk_input=False,
                            duplicate_sample=True,
                            append_depth=False,
                            gs_cam_format=False,
                            orthog_duplicate=False,
                            **kwargs):

    post_process_cls = PostProcess(
        reso,
        reso_encoder,
        imgnet_normalize=imgnet_normalize,
        plucker_embedding=plucker_embedding,
        decode_encode_img_only=decode_encode_img_only,
        mv_input=mv_input,
        split_chunk_input=split_chunk_input,
        duplicate_sample=duplicate_sample,
        append_depth=append_depth,
        gs_cam_format=gs_cam_format,
        orthog_duplicate=orthog_duplicate,
    )

    if isinstance(file_path, list):
        all_shards = []
        for url_path in file_path:
            all_shards.extend(wds.shardlists.expand_source(url_path))
        logger.log('all_shards', all_shards)
    else:
        all_shards = file_path
    if not load_instance:
        if not split_chunk_input:
            dataset = wds.DataPipeline(
                wds.ResampledShards(all_shards),
                wds.shuffle(50),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.shuffle(1000),
                wds.decode(wds.autodecode.basichandlers),
                wds.to_tuple("sample.pyd"),
                wds.map(post_process_cls.decode_zip),
                wds.map(post_process_cls.paired_post_process),
                wds.batched(16, partial=True)
            )

        elif load_gzip:
            dataset = wds.DataPipeline(
                wds.ResampledShards(all_shards),
                wds.shuffle(10),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.decode('rgb8'),
                wds.to_tuple('raw_img.png', 'depth_alpha.png'),
                wds.map(post_process_cls.decode_gzip),
                wds.batched(20, partial=True)
            )

        else:
            dataset = wds.DataPipeline(
                wds.ResampledShards(all_shards),
                wds.shuffle(100),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.shuffle(4000 // split_chunk_size),
                wds.decode(wds.autodecode.basichandlers),
                wds.to_tuple("sample.pyd"),
                wds.map(post_process_cls.decode_zip),
                wds.map(post_process_cls.paired_post_process_chunk),
                wds.batched(120 // split_chunk_size, partial=True)
            )

        loader_shard = wds.WebLoader(
            dataset,
            num_workers=num_workers,
            drop_last=False,
            batch_size=None,
            shuffle=False,
            persistent_workers=num_workers > 0).unbatched().shuffle(
                1000 // split_chunk_size).batched(batch_size).map(
                    post_process_cls.create_dict)

        if mv_input:
            loader_shard = loader_shard.map(post_process_cls.prepare_mv_input)

    else:
        assert batch_size == 1

        dataset = wds.DataPipeline(
            wds.ResampledShards(all_shards),
            wds.shuffle(50),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.detshuffle(100),
            wds.decode(wds.autodecode.basichandlers),
            wds.to_tuple("sample.pyd"),
            wds.map(post_process_cls.decode_zip),
            wds.map(post_process_cls._post_process_batch_sample),
            wds.batched(2, partial=True)
        )

        loader_shard = wds.WebLoader(
            dataset,
            num_workers=num_workers,
            drop_last=False,
            batch_size=None,
            shuffle=False,
            persistent_workers=num_workers > 0).unbatched().shuffle(200).batched(batch_size).map(
                post_process_cls.single_instance_sample_create_dict)

    return loader_shard


class PostProcessForDiff:
    def __init__(
        self,
        reso,
        reso_encoder,
        imgnet_normalize,
        plucker_embedding,
        decode_encode_img_only,
        mv_latent_dir,
    ) -> None:
        self.plucker_embedding = plucker_embedding
        self.mv_latent_dir = mv_latent_dir
        self.decode_encode_img_only = decode_encode_img_only

        transformations = [
            transforms.ToTensor(),
        ]
        if imgnet_normalize:
            transformations.append(
                transforms.Normalize((0.485, 0.456, 0.406),
                                        (0.229, 0.224, 0.225))
            )
        else:
            transformations.append(
                transforms.Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5))
            )

        self.normalize = transforms.Compose(transformations)

        self.reso_encoder = reso_encoder
        self.reso = reso
        self.instance_data_length = 40
        self.pair_per_instance = 2
        self.camera = torch.load('eval_pose.pt', map_location='cpu').numpy()
        self.canonical_frame = self.camera[25:26]
        self.canonical_frame_pos = self.canonical_frame[:, :16].reshape(4, 4)

    def get_rays_kiui(self, c, opengl=True):
        h, w = self.reso_encoder, self.reso_encoder
        intrinsics, pose = c[16:], c[:16].reshape(4, 4)
        fx = fy = 525
        cx = cy = 256
        factor = self.reso / (cx * 2)
        fx = fx * factor
        fy = fy * factor

        x, y = torch.meshgrid(
            torch.arange(w, device=pose.device),
            torch.arange(h, device=pose.device),
            indexing="xy",
        )
        x = x.flatten()
        y = y.flatten()

        cx = w * 0.5
        cy = h * 0.5

        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - cx + 0.5) / fx,
                    (y - cy + 0.5) / fy * (-1.0 if opengl else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if opengl else 1.0),
        )

        rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)
        rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d)

        rays_o = rays_o.view(h, w, 3)
        rays_d = safe_normalize(rays_d).view(h, w, 3)

        return rays_o, rays_d

    def gen_rays(self, c):
        intrinsics, c2w = c[16:], c[:16].reshape(4, 4)
        self.h = self.reso_encoder
        self.w = self.reso_encoder
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32) + 0.5,
            torch.arange(self.w, dtype=torch.float32) + 0.5,
            indexing='ij')

        yy = yy / self.h
        xx = xx / self.w

        cx, cy, fx, fy = intrinsics[2], intrinsics[5], intrinsics[0], intrinsics[4]
        c2w = torch.from_numpy(c2w).float()

        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)
        del xx, yy, zz
        dirs = (c2w[None, :3, :3] @ dirs)[..., 0]

        origins = c2w[None, :3, 3].expand(self.h * self.w, -1).contiguous()
        origins = origins.view(self.h, self.w, 3)
        dirs = dirs.view(self.h, self.w, 3)

        return origins, dirs

    def normalize_camera(self, c):
        c = c[None]
        B = c.shape[0]

        camera_poses = c[:, :16].reshape(B, 4, 4)

        cam_radius = np.linalg.norm(
            self.canonical_frame_pos.reshape(4, 4)[:3, 3],
            axis=-1,
            keepdims=False)
        frame1_fixed_pos = np.eye(4)
        frame1_fixed_pos[2, -1] = -cam_radius

        transform = frame1_fixed_pos @ np.linalg.inv(
            self.canonical_frame_pos)

        new_camera_poses = transform[None] @ camera_poses

        c = np.concatenate([new_camera_poses.reshape(B, 16), c[:, 16:]],
                        axis=-1)

        return c[0]

    def _post_process_sample(self, data_sample):
        raw_img, c, caption, ins = data_sample

        img = raw_img

        img = torch.from_numpy(img).permute(2, 0, 1) / 127.5 - 1

        latent_path = Path(self.mv_latent_dir.replace('v=4-final', 'v=4-rotate'), ins, 'latent.npy')

        latent = np.load(latent_path)

        return (latent, img, c, caption, ins)

    def rand_sample_idx(self):
        return random.randint(0, self.instance_data_length - 1)

    def rand_pair(self):
        return (self.rand_sample_idx() for _ in range(2))

    def paired_post_process(self, sample):
        all_inp_list = []
        all_nv_list = []
        caption, ins = sample[-2:]
        for _ in range(self.pair_per_instance):
            cano_idx, nv_idx = self.rand_pair()
            cano_sample = self._post_process_sample(
                item[cano_idx] for item in sample[:-2])
            nv_sample = self._post_process_sample(item[nv_idx]
                                                    for item in sample[:-2])
            all_inp_list.extend(cano_sample)
            all_nv_list.extend(nv_sample)
        return (*all_inp_list, *all_nv_list, caption, ins)

    def single_sample_create_dict(self, sample, prefix=''):
        latent, img, c, caption, ins = sample
        return {
            'latent': latent,
            'img': img,
            'c': c,
            'caption': caption,
            'ins': ins
        }

    def decode_zip(self, sample_pyd, shape=(256, 256)):
        if isinstance(sample_pyd, tuple):
            sample_pyd = sample_pyd[0]
        assert isinstance(sample_pyd, dict)

        raw_img = decompress_and_open_image_gzip(
            sample_pyd['raw_img'],
            is_img=True,
            decompress=True,
            decompress_fn=lz4.frame.decompress)

        caption = sample_pyd['caption'].decode('utf-8')
        ins = sample_pyd['ins'].decode('utf-8')

        c = decompress_array(sample_pyd['c'], (25, ),
                                np.float32,
                                decompress=True,
                                decompress_fn=lz4.frame.decompress)

        return raw_img, c, caption, ins

    def create_dict(self, sample):
        cano_sample = self.single_sample_create_dict(sample, prefix='')
        return cano_sample


def load_wds_diff_ResampledShard(file_path,
                                 batch_size,
                                 num_workers,
                                 reso,
                                 reso_encoder,
                                 test=False,
                                 preprocess=None,
                                 imgnet_normalize=True,
                                 plucker_embedding=False,
                                 decode_encode_img_only=False,
                                 mv_latent_dir='',
                                 **kwargs):

    post_process_cls = PostProcessForDiff(
        reso,
        reso_encoder,
        imgnet_normalize=imgnet_normalize,
        plucker_embedding=plucker_embedding,
        decode_encode_img_only=decode_encode_img_only,
        mv_latent_dir=mv_latent_dir,
    )

    if isinstance(file_path, list):
        all_shards = []
        for url_path in file_path:
            all_shards.extend(wds.shardlists.expand_source(url_path))
        logger.log('all_shards', all_shards)
    else:
        all_shards = file_path

    dataset = wds.DataPipeline(
        wds.ResampledShards(all_shards),
        wds.shuffle(100),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.shuffle(20000),
        wds.decode(wds.autodecode.basichandlers),
        wds.to_tuple("sample.pyd"),
        wds.map(post_process_cls.decode_zip),
        wds.map(post_process_cls._post_process_sample),
        wds.batched(100, partial=True)
    )

    loader_shard = wds.WebLoader(
        dataset,
        num_workers=num_workers,
        drop_last=False,
        batch_size=None,
        shuffle=False,
        persistent_workers=num_workers > 0).unbatched().shuffle(2500).batched(batch_size).map(
            post_process_cls.create_dict)

    return loader_shard


def load_wds_data(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        num_workers=6,
        plucker_embedding=False,
        decode_encode_img_only=False,
        load_wds_diff=False,
        load_wds_latent=False,
        load_instance=False,
        mv_input=False,
        split_chunk_input=False,
        duplicate_sample=True,
        mv_latent_dir='',
        append_depth=False,
        gs_cam_format=False,
        orthog_duplicate=False,
        **args):

    if load_wds_diff:
        wds_loader = load_wds_diff_ResampledShard(
            file_path,
            batch_size=batch_size,
            num_workers=num_workers,
            reso=reso,
            reso_encoder=reso_encoder,
            plucker_embedding=plucker_embedding,
            decode_encode_img_only=decode_encode_img_only,
            mv_input=mv_input,
            split_chunk_input=split_chunk_input,
            append_depth=append_depth,
            mv_latent_dir=mv_latent_dir,
            gs_cam_format=gs_cam_format,
            orthog_duplicate=orthog_duplicate,
        )
    elif load_wds_latent:
        wds_loader = load_wds_latent_ResampledShard(
            file_path,
            batch_size=batch_size,
            num_workers=num_workers,
            reso=reso,
            reso_encoder=reso_encoder,
            plucker_embedding=plucker_embedding,
            decode_encode_img_only=decode_encode_img_only,
            mv_input=mv_input,
            split_chunk_input=split_chunk_input,
        )

    else:
        wds_loader = load_wds_ResampledShard(
            file_path,
            batch_size=batch_size,
            num_workers=num_workers,
            reso=reso,
            reso_encoder=reso_encoder,
            plucker_embedding=plucker_embedding,
            decode_encode_img_only=decode_encode_img_only,
            load_instance=load_instance,
            mv_input=mv_input,
            split_chunk_input=split_chunk_input,
            duplicate_sample=duplicate_sample,
            append_depth=append_depth,
            gs_cam_format=gs_cam_format,
            orthog_duplicate=orthog_duplicate,
        )

    while True:
        yield from wds_loader


class PostProcess_forlatent:
    def __init__(
        self,
        reso,
        reso_encoder,
        imgnet_normalize,
        plucker_embedding,
        decode_encode_img_only,
    ) -> None:
        self.plucker_embedding = plucker_embedding
        self.decode_encode_img_only = decode_encode_img_only

        transformations = [
            transforms.ToTensor(),
        ]
        if imgnet_normalize:
            transformations.append(
                transforms.Normalize((0.485, 0.456, 0.406),
                                        (0.229, 0.224, 0.225))
            )
        else:
            transformations.append(
                transforms.Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5))
            )

        self.normalize = transforms.Compose(transformations)

        self.reso_encoder = reso_encoder
        self.reso = reso
        self.instance_data_length = 40
        self.pair_per_instance = 2

    def _post_process_sample(self, data_sample):
        raw_img, c, caption, ins = data_sample

        if raw_img.shape[-2] != self.reso_encoder:
            img_to_encoder = cv2.resize(
                raw_img, (self.reso_encoder, self.reso_encoder),
                interpolation=cv2.INTER_LANCZOS4)
        else:
            img_to_encoder = raw_img

        img_to_encoder = self.normalize(img_to_encoder)
        if self.plucker_embedding:
            rays_o, rays_d = self.gen_rays(c)
            rays_plucker = torch.cat(
                [torch.cross(rays_o, rays_d, dim=-1), rays_d],
                dim=-1).permute(2, 0, 1)
            img_to_encoder = torch.cat([img_to_encoder, rays_plucker], 0)

        img = cv2.resize(raw_img, (self.reso, self.reso),
                            interpolation=cv2.INTER_LANCZOS4)

        img = torch.from_numpy(img).permute(2, 0, 1) / 127.5 - 1

        return (img_to_encoder, img, c, caption, ins)

    def rand_sample_idx(self):
        return random.randint(0, self.instance_data_length - 1)

    def rand_pair(self):
        return (self.rand_sample_idx() for _ in range(2))

    def paired_post_process(self, sample):
        all_inp_list = []
        all_nv_list = []
        caption, ins = sample[-2:]
        for _ in range(self.pair_per_instance):
            cano_idx, nv_idx = self.rand_pair()
            cano_sample = self._post_process_sample(
                item[cano_idx] for item in sample[:-2])
            nv_sample = self._post_process_sample(item[nv_idx]
                                                    for item in sample[:-2])
            all_inp_list.extend(cano_sample)
            all_nv_list.extend(nv_sample)
        return (*all_inp_list, *all_nv_list, caption, ins)
        # return [cano_sample, nv_sample, caption, ins]
        # return (*cano_sample, *nv_sample, caption, ins)
    def paired_post_process(self, sample):
        # repeat n times?
        all_inp_list = []
        all_nv_list = []
        caption, ins = sample[-2:]
        # expanded_return = []
        for _ in range(self.pair_per_instance):
            cano_idx, nv_idx = self.rand_pair()
            cano_sample = self._post_process_sample(
                item[cano_idx] for item in sample[:-2])
            nv_sample = self._post_process_sample(item[nv_idx]
                                                    for item in sample[:-2])
            all_inp_list.extend(cano_sample)
            all_nv_list.extend(nv_sample)
        return (*all_inp_list, *all_nv_list, caption, ins)

    def single_sample_create_dict(self, sample, prefix=''):
        img_to_encoder, img, c, caption, ins = sample
        return {
            'img_to_encoder': img_to_encoder,
            'img': img,
            'c': c,
            'caption': caption,
            'ins': ins
        }

    def decode_zip(self, sample_pyd, shape=(256, 256)):
        if isinstance(sample_pyd, tuple):
            sample_pyd = sample_pyd[0]
        assert isinstance(sample_pyd, dict)

        latent = sample_pyd['latent']
        caption = sample_pyd['caption'].decode('utf-8')
        c = sample_pyd['c']

        return latent, caption, c

    def create_dict(self, sample):
        return {
            'latent': sample[0],
            'caption': sample[1],
            'c': sample[2],
        }


def load_wds_latent_ResampledShard(file_path,
                                   batch_size,
                                   num_workers,
                                   reso,
                                   reso_encoder,
                                   test=False,
                                   preprocess=None,
                                   imgnet_normalize=True,
                                   plucker_embedding=False,
                                   decode_encode_img_only=False,
                                   **kwargs):

    post_process_cls = PostProcess_forlatent(
        reso,
        reso_encoder,
        imgnet_normalize=imgnet_normalize,
        plucker_embedding=plucker_embedding,
        decode_encode_img_only=decode_encode_img_only,
    )

    if isinstance(file_path, list):
        all_shards = []
        for url_path in file_path:
            all_shards.extend(wds.shardlists.expand_source(url_path))
        logger.log('all_shards', all_shards)
    else:
        all_shards = file_path

    dataset = wds.DataPipeline(
        wds.ResampledShards(all_shards),
        wds.shuffle(50),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.detshuffle(2500),
        wds.decode(wds.autodecode.basichandlers),
        wds.to_tuple("sample.pyd"),
        wds.map(post_process_cls.decode_zip),
        wds.batched(150, partial=True)
    )

    loader_shard = wds.WebLoader(
        dataset,
        num_workers=num_workers,
        drop_last=False,
        batch_size=None,
        shuffle=False,
        persistent_workers=num_workers > 0).unbatched().shuffle(1000).batched(batch_size).map(
            post_process_cls.create_dict)

    return loader_shard
