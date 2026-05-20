import os
import math
import time
from typing import Any
import cv2
cv2.setNumThreads(0) # disable multiprocess
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from pdb import set_trace as st
from pathlib import Path

from einops import rearrange
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import lz4.frame
from nsr.volumetric_rendering.ray_sampler import RaySampler
import point_cloud_utils as pcu

import torch.multiprocessing

from utils.general_utils import matrix_to_quaternion

from guided_diffusion import logger
import json

from utils.gs_utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def resize_depth_mask(depth_to_resize, resolution):
    depth_resized = cv2.resize(depth_to_resize, (resolution, resolution),
                               interpolation=cv2.INTER_LANCZOS4)
    return depth_resized, depth_resized > 0  # type: ignore

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

    return depth_resized, depth_resized > 0  # type: ignore

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
        if pcd_path is None:  
            self.pcd_path = Path('path/to/pcd')

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
                                     (0.229, 0.224, 0.225))  
            )
        else:
            transformations.append(
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5)))

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

    def _post_process_batch_sample(self, sample):
        caption, ins = sample[-2:]
        instance_samples = []

        for instance_idx in range(sample[0].shape[0]):
            instance_samples.append(
                self._post_process_sample(item[instance_idx]
                                          for item in sample[:-2]))

        return (*instance_samples, caption, ins)

    def _post_process_sample(self, data_sample):
        raw_img, depth, c, bbox = data_sample

        bbox = (bbox * (self.reso / 256)).astype(np.uint8)

        if raw_img.shape[-2] != self.reso_encoder:
            img_to_encoder = cv2.resize(raw_img,
                                        (self.reso_encoder, self.reso_encoder),
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

        if self.decode_encode_img_only:
            depth_reso, fg_mask_reso = depth, depth
        else:
            depth_reso, fg_mask_reso = resize_depth_mask(depth, self.reso)

        return (img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox)

    def canonicalize_pts(self, c, pcd, for_encoder=True, canonical_idx=0):
        assert c.shape[0] == self.chunk_size
        assert for_encoder

        B = c.shape[0]
        camera_poses = c[:, :16].reshape(B, 4, 4)

        cam_radius = np.linalg.norm(
            c[[0, self.V]][:, :16].reshape(2, 4, 4)[:, :3, 3],
            axis=-1,
            keepdims=False)
        frame1_fixed_pos = np.repeat(np.eye(4)[None], 2, axis=0)
        frame1_fixed_pos[:, 2, -1] = -cam_radius

        transform = frame1_fixed_pos @ np.linalg.inv(camera_poses[[0, self.V]])
        transform = np.expand_dims(transform, axis=1)

        repeated_homo_pcd = np.repeat(np.concatenate(
            [pcd, np.ones_like(pcd[..., 0:1])], -1)[None],
                                      2,
                                      axis=0)[..., None]
        new_pcd = (transform @ repeated_homo_pcd)[..., :3, 0]

        return new_pcd

    def normalize_camera(self, c, for_encoder=True, canonical_idx=0):
        assert c.shape[0] == self.chunk_size

        B = c.shape[0]
        camera_poses = c[:, :16].reshape(B, 4, 4)

        if for_encoder:
            encoder_canonical_idx = [0, self.V]
            cam_radius = np.linalg.norm(
                c[encoder_canonical_idx][:, :16].reshape(2, 4, 4)[:, :3, 3],
                axis=-1,
                keepdims=False)
            frame1_fixed_pos = np.repeat(np.eye(4)[None], 2, axis=0)
            frame1_fixed_pos[:, 2, -1] = -cam_radius

            transform = frame1_fixed_pos @ np.linalg.inv(
                camera_poses[encoder_canonical_idx])

            new_camera_poses = np.repeat(
                transform, self.V, axis=0
            ) @ camera_poses

        else:
            cam_radius = np.linalg.norm(
                c[canonical_idx][:16].reshape(4, 4)[:3, 3],
                axis=-1,
                keepdims=False)
            frame1_fixed_pos = np.eye(4)
            frame1_fixed_pos[2, -1] = -cam_radius

            transform = frame1_fixed_pos @ np.linalg.inv(
                camera_poses[canonical_idx])

            new_camera_poses = np.repeat(transform[None],
                                         self.chunk_size,
                                         axis=0) @ camera_poses

        c = np.concatenate([new_camera_poses.reshape(B, 16), c[:, 16:]],
                           axis=-1)

        return c

    def get_plucker_ray(self, c):
        rays_plucker = []
        for idx in range(c.shape[0]):
            rays_o, rays_d = self.gen_rays(c[idx])
            rays_plucker.append(
                torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d],
                          dim=-1).permute(2, 0, 1))
        rays_plucker = torch.stack(rays_plucker, 0)
        return rays_plucker

    def _post_process_sample_batch(self, data_sample):
        alpha = None
        if len(data_sample) == 4:
            raw_img, depth, c, bbox = data_sample
        else:
            raw_img, depth, c, alpha, bbox = data_sample

        if isinstance(depth, tuple):
            # self.append_normal = True
            self.append_normal = False
            depth, normal = depth
        else:
            self.append_normal = False
            normal = None

        depth_reso, fg_mask_reso = resize_depth_mask_Tensor(
            torch.from_numpy(depth), self.reso)

        if alpha is None:
            alpha = fg_mask_reso
        else:
            alpha = torch.from_numpy(alpha/255.0).float()
            if alpha.shape[-1] != self.reso:
                alpha = torch.nn.functional.interpolate(
                    input=alpha.unsqueeze(1),
                    size=(self.reso, self.reso),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(1)

        if self.reso < 256:
            bbox = (bbox * (self.reso / 256)).astype(np.uint8)
        else:
            bbox = bbox.astype(np.uint8)

        assert raw_img.shape[-2] == self.reso_encoder

        raw_img = torch.from_numpy(raw_img).permute(0, 3, 1,
                                                    2) / 255.0
        img_to_encoder = self.normalize(raw_img)

        if raw_img.shape[-1] != self.reso:
            img = torch.nn.functional.interpolate(
                input=raw_img,
                size=(self.reso, self.reso),
                mode='bilinear',
                align_corners=False,
            ) * 2 - 1
        else:
            img = raw_img * 2 - 1

        if self.frame_0_as_canonical:
            encoder_canonical_idx = [0, self.V]

            c_for_encoder = self.normalize_camera(c, for_encoder=True)
            c_for_render = self.normalize_camera(
                c,
                for_encoder=False,
                canonical_idx=encoder_canonical_idx[0]
            )
            c_for_render_nv = self.normalize_camera(
                c,
                for_encoder=False,
                canonical_idx=encoder_canonical_idx[1]
            )
            c_for_render = np.concatenate([c_for_render, c_for_render_nv],
                                          axis=-1)

        else:
            c_for_encoder, c_for_render = c, c

        if self.plucker_embedding:
            rays_plucker = self.get_plucker_ray(c_for_encoder)
            img_to_encoder = torch.cat([img_to_encoder, rays_plucker],
                                       1)

        if self.append_depth:
            normalized_depth = torch.from_numpy(depth).clone().unsqueeze(1)
            img_to_encoder = torch.cat([img_to_encoder, normalized_depth],
                                       1)

        c = torch.from_numpy(c_for_render).to(torch.float32)

        return (img_to_encoder, img, alpha, depth_reso, c,
                torch.from_numpy(bbox))

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
            cano_sample = self._post_process_sample(item[cano_idx]
                                                    for item in sample[:-2])
            nv_sample = self._post_process_sample(item[nv_idx]
                                                  for item in sample[:-2])
            all_inp_list.extend(cano_sample)
            all_nv_list.extend(nv_sample)
        return (*all_inp_list, *all_nv_list, caption, ins)

    def get_source_cw2wT(self, source_cameras_view_to_world):
        return matrix_to_quaternion(
            source_cameras_view_to_world[:3, :3].transpose(0, 1))

    def c_to_3dgs_format(self, pose):
        c2w = pose[:16].reshape(4, 4)

        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        fx = pose[16]
        FovX = focal2fov(fx, 1)
        FovY = focal2fov(fx, 1)

        tanfovx = math.tan(FovX * 0.5)
        tanfovy = math.tan(FovY * 0.5)

        assert tanfovx == tanfovy

        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0

        view_world_transform = torch.tensor(getView2World(R, T, trans,
                                                          scale)).transpose(0, 1)

        world_view_transform = torch.tensor(getWorld2View2(R, T, trans,
                                                           scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear,
                                                zfar=self.zfar,
                                                fovX=FovX,
                                                fovY=FovY).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
            projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

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

    def paired_post_process_chunk(self, sample):
        all_inp_list = []
        all_nv_list = []
        auxiliary_sample = list(sample[-2:])
        ins = sample[-1]

        assert sample[0].shape[0] == self.chunk_size

        if self.load_pcd:
            fps_pcd = pcu.load_mesh_v(str(self.pcd_path / ins /
                                          'fps-4096.ply'))

            auxiliary_sample += [fps_pcd]

        assert self.duplicate_sample
        if self.duplicate_sample:
            if self.chunk_size in [20, 12]:
                shuffle_sample = sample[:-2]
            else:
                shuffle_sample = []
                indices = np.random.permutation(self.chunk_size)
                for _, item in enumerate(sample[:-2]):
                    shuffle_sample.append(item[indices])

            processed_sample = self._post_process_sample_batch(shuffle_sample)

            if self.load_pcd:  
                if self.frame_0_as_canonical:
                    auxiliary_sample[-1] = self.canonicalize_pts( c=shuffle_sample[2], pcd=auxiliary_sample[-1], for_encoder=True)
                else:
                    auxiliary_sample[-1] = np.repeat(auxiliary_sample[-1][None], 2, axis=0)

            assert not self.orthog_duplicate

            all_inp_list.extend(item[:self.V] for item in processed_sample)
            all_nv_list.extend(item[self.V:] for item in processed_sample)

            return (*all_inp_list, *all_nv_list, *auxiliary_sample)

        else:
            processed_sample = self._post_process_sample_batch(
                item[:4] for item in sample[:-2])

            all_inp_list.extend(item for item in processed_sample)
            all_nv_list.extend(item
                               for item in processed_sample)

        return (*all_inp_list, *all_nv_list, *auxiliary_sample)

    def single_sample_create_dict_noBatch(self, sample, prefix=''):
        img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox = sample

        if self.gs_cam_format:
            B, V, _ = c.shape
            c = rearrange(c, 'B V C -> (B V) C').cpu().numpy()
            all_gs_c = [self.c_to_3dgs_format(pose) for pose in c]
            c = {
                k: rearrange(torch.stack([gs_c[k] for gs_c in all_gs_c]),
                             '(B V) ... -> B V ...',
                             B=B,
                             V=V)
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
                k: rearrange(torch.stack([gs_c[k] for gs_c in all_gs_c]),
                             '(B V) ... -> B V ...',
                             B=B,
                             V=V)
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
            prefix='')

        return {
            **inp_sample,
            'caption': sample[-2],
            'ins': sample[-1]
        }

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


def chunk_collate_fn(sample):
    sample = torch.utils.data.default_collate(sample)
    if 'caption' in sample:
        bs = len(sample['caption'])  # number of instances
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
                    sample[k] = v[0].float().item() # tanfov.
    if 'c' in sample:
        if isinstance(sample['c'], dict): # 3dgs
            merge_internal_batch(sample['c'], merge_b_only=True)
            merge_internal_batch(sample['nv_c'], merge_b_only=True)

    merge_internal_batch(sample)

    return sample

def load_data_3D_VAR(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        num_workers=1,
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
        text_conditioned=False,
        **kwargs):
    """
    Load 3D data with various dataset formats and configurations.
    
    Args:
        file_path: Path to dataset
        reso: Output resolution
        reso_encoder: Encoder input resolution
        batch_size: Batch size for dataloader
        num_workers: Number of dataloader workers
        load_depth: Whether to load depth maps
        preprocess: Preprocessing function
        imgnet_normalize: Whether to normalize with ImageNet stats
        dataset_size: Size limit for dataset (-1 for no limit)
        trainer_name: Name of trainer ('input_rec' or 'nv')
        use_lmdb: Whether to use LMDB dataset
        use_lmdb_compressed: Whether to use compressed LMDB
        infi_sampler: Whether to use infinite sampler
        eval: Whether in evaluation mode
        load_whole: Whether to load whole dataset
    """

    collate_fn = None

    # Select dataset class based on configuration
    if use_lmdb:
        logger.log('Using LMDB dataset')
        raise NotImplementedError
    else:
        if eval:
            dataset_cls = ChunkObjaverseDataset_eval
            collate_fn = chunk_collate_fn
        else:
            dataset_cls = ChunkObjaverseDataset
            collate_fn = chunk_collate_fn

    # Initialize dataset
    dataset = dataset_cls(
        file_path,
        reso,
        reso_encoder,
        test=False,
        preprocess=preprocess,
        load_depth=load_depth,
        imgnet_normalize=imgnet_normalize,
        dataset_size=dataset_size,
        load_whole=load_whole,
        text_conditioned=text_conditioned,
        **kwargs
    )

    print(f'Dataset class: {trainer_name}, size: {len(dataset)}')

    # Create data loader with infinite sampler if requested
    if infi_sampler:
        train_sampler = DistributedSampler(
            dataset=dataset,
            shuffle=True,
            drop_last=True
        )

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

def get_intri(target_im=None, h=None, w=None, normalize=False):
    """
    Get camera intrinsic matrix.
    
    Args:
        target_im: Target image (optional)
        h: Image height (required if target_im not provided)
        w: Image width (required if target_im not provided) 
        normalize: Whether to normalize by image height
    """
    if target_im is None:
        assert h is not None and w is not None
    else:
        h, w = target_im.shape[:2]

    # Calculate focal length and construct intrinsic matrix
    fx = fy = 1422.222
    res_raw = 1024
    f_x = f_y = fx * h / res_raw
    K = np.array([f_x, 0, w/2, 0, f_y, h/2, 0, 0, 1]).reshape(3, 3)
    
    if normalize:
        K[:6] /= h
        
    return K

def current_milli_time():
    """Get current time in milliseconds"""
    return round(time.time() * 1000)

class ChunkObjaverseDataset(Dataset):
    """Dataset class for loading and processing Objaverse data in chunks.
    
    Loads multi-view images and metadata from Objaverse dataset, processes them into
    chunks for training. Supports loading latent codes and various data augmentations.
    """
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
            text_conditioned=False,
            **kwargs):
        super().__init__()

        # Basic configurations
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
        self.load_whole = load_whole
        self.text_conditioned = text_conditioned
        # Get camera intrinsics
        self.intrinsics = get_intri(h=self.reso, w=self.reso, normalize=True).reshape(9)
        assert not self.classes, "Class conditioning not supported yet."

        # Dataset metadata
        self.dataset_name = Path(self.file_path).stem.split('_')[0]
        self.zfar = 100.0
        self.znear = 0.01
        self.chunk_list = []
        self.img_ext = 'png'

        # Load dataset json
        dataset_json = []
        categories = [
                    'Furnitures',
                    'daily-used',
                    'Animals',
                    'Food',
                    'Plants',
                    'Electronics',
                    'BuildingsOutdoor',
                    'Transportations_tar',
                    'Human-Shape',]
        for category in categories:
            with open(f'{self.file_path}/dataset.json', 'r') as f:
                category_data = json.load(f)[category][:-100]
                dataset_json.extend(category_data)

        # Build chunk list
        for item in dataset_json:
            self.chunk_list.append(item)

        # Initialize post-processing
        self.post_process = PostProcess(
            reso=reso,
            reso_encoder=reso_encoder,
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

    def read_chunk(self, chunk_path):
        """Read and process a data chunk from disk.
        
        Args:
            chunk_path: Path to the chunk directory
            
        Returns:
            Tuple containing processed images, depth maps, camera params, etc.
        """
        # Load and reshape raw image
        raw_img = imageio.imread(os.path.join(chunk_path, f'raw_img.{self.img_ext}'))
        h, bw, c = raw_img.shape
        raw_img = raw_img.reshape(h, self.chunk_size, -1, c).transpose(1, 0, 2, 3)

        # Load and process depth and alpha
        depth_alpha = imageio.imread(os.path.join(chunk_path, 'depth_alpha.jpg'))
        depth_alpha = depth_alpha.reshape(h * 2, self.chunk_size, -1).transpose(1, 0, 2)
        depth, alpha = np.split(depth_alpha, 2, axis=1)

        # Load camera parameters and bounding boxes
        c = np.load(os.path.join(chunk_path, 'c.npy'))
        d_near_far = np.load(os.path.join(chunk_path, 'd_near_far.npy'))
        bbox = np.load(os.path.join(chunk_path, 'bbox.npy'))

        # Process depth values
        d_near = d_near_far[0].reshape(self.chunk_size, 1, 1)
        d_far = d_near_far[1].reshape(self.chunk_size, 1, 1)
        depth = 1 / ((depth / 255) * (d_far - d_near) + d_near)
        depth[depth > 2.9] = 0.0  # Set background depth to 0

        # Load text metadata
        with open(os.path.join(chunk_path, 'caption_3dtopia.txt'), 'r', encoding="utf-8") as f:
            caption = f.read()
        with open(os.path.join(chunk_path, 'ins.txt'), 'r', encoding="utf-8") as f:
            ins = f.read()

        return raw_img, depth, c, alpha, bbox, caption, ins

    def load_latent(self, sample, latent_path):
        """Load pre-computed latent codes and embeddings.
        
        Args:
            sample: Current sample dict to update
            latent_path: Path to latent code directory
            
        Returns:
            Updated sample dict with latent codes
        """
        # Load latent codes and embeddings
        gt_BL = torch.from_numpy(np.load(os.path.join(latent_path, "gt_BL_dim_8_l2norm_lrm_256.npy")))
        x_BLCv_wo_first_l = torch.from_numpy(np.load(os.path.join(latent_path, "x_BLCv_wo_first_l_dim_8_l2_norm_lrm_256.npy")))
        if self.text_conditioned:
            text_embedding = torch.from_numpy(np.load(os.path.join(latent_path, "text_embedding_lrm_3dtopia.npy")))
            text_pooler_ouput = torch.from_numpy(np.load(os.path.join(latent_path, "text_pooler_output_lrm_3dtopia.npy")))

            sample.update({
                'gt_BL': gt_BL,
                'x_BLCv_wo_first_l': x_BLCv_wo_first_l,
                'text_embedding': text_embedding,
                'text_pooler_output': text_pooler_ouput})
        else:
            image_embedding = torch.from_numpy(np.load(os.path.join(latent_path, "image_dino_embedding_lrm.npy")))
            
            # Split DINO embeddings
            image_dino_embedding = image_embedding[1:, :]
            image_dino_pooler_output = image_embedding[0]

            # Update sample dict
            sample.update({
                'gt_BL': gt_BL,
                'x_BLCv_wo_first_l': x_BLCv_wo_first_l,
                'image_dino_pooler_output': image_dino_pooler_output,
                'image_dino_embedding': image_dino_embedding,
            })
            
        return sample   

    def __len__(self):
        return len(self.chunk_list)

    def __getitem__(self, index):
        """Get a processed data sample.
        
        Args:
            index: Sample index
            
        Returns:
            Dict containing processed images, latents and metadata
        """
        sample = {}
        chunk_path = os.path.join(self.file_path, self.chunk_list[index])
        
        if self.load_whole:
            raw_sample = self.read_chunk(chunk_path)
            sample = self.post_process.paired_post_process_chunk(raw_sample)
            sample = self.post_process.create_dict_nobatch(sample)

        sample['sample_path'] = chunk_path
        sample = self.load_latent(sample, chunk_path)
        
        return sample


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
            text_conditioned=False,
            **kwargs):

        super().__init__()

        # Basic configurations
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
        self.load_whole = load_whole
        self.text_conditioned = text_conditioned    

        # Get camera intrinsics
        self.intrinsics = get_intri(h=self.reso, w=self.reso, normalize=True).reshape(9)
        assert not self.classes, "Class conditioning not supported"

        # Dataset metadata
        self.dataset_name = Path(self.file_path).stem.split('_')[0]
        self.zfar = 100.0
        self.znear = 0.01
        self.chunk_list = []
        self.img_ext = 'png'

        # Load dataset json
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
                    'Human-Shape',]:
            with open(f'{self.file_path}/dataset.json', 'r') as f:
                cl_dataset_json = json.load(f)[cl][-100:]
            dataset_json.extend(cl_dataset_json)

        # Build chunk list
        for v in dataset_json:
            self.chunk_list.append(v)

        # Initialize post processor
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
        return []

    def read_chunk(self, chunk_path):
        # Load raw image and reshape
        raw_img = imageio.imread(os.path.join(chunk_path, f'raw_img.{self.img_ext}'))
        h, bw, c = raw_img.shape
        raw_img = raw_img.reshape(h, self.chunk_size, -1, c).transpose(
            (1, 0, 2, 3))

        # Load depth and alpha
        depth_alpha = imageio.imread(os.path.join(chunk_path, 'depth_alpha.jpg'))
        depth_alpha = depth_alpha.reshape(h * 2, self.chunk_size, -1).transpose((1, 0, 2))
        depth, alpha = np.split(depth_alpha, 2, axis=1)

        # Load camera params and bounding box
        c = np.load(os.path.join(chunk_path, 'c.npy'))
        d_near_far = np.load(os.path.join(chunk_path, 'd_near_far.npy'))
        bbox = np.load(os.path.join(chunk_path, 'bbox.npy'))

        # Process depth
        d_near = d_near_far[0].reshape(self.chunk_size, 1, 1)
        d_far = d_near_far[1].reshape(self.chunk_size, 1, 1)
        depth = 1 / ((depth / 255) * (d_far - d_near) + d_near)
        depth[depth > 2.9] = 0.0

        # Load caption and instance ID
        with open(os.path.join(chunk_path, 'caption_3dtopia.txt'), 'r', encoding="utf-8") as f:
            caption = f.read()
        with open(os.path.join(chunk_path, 'ins.txt'), 'r', encoding="utf-8") as f:
            ins = f.read()

        return raw_img, depth, c, alpha, bbox, caption, ins

    def load_latent(self, sample, latent_path=None):
        if os.path.exists(os.path.join(latent_path, "gt_BL_dim_8_l2norm_lrm_256.npy")):
            gt_BL = torch.from_numpy(np.load(os.path.join(latent_path, "gt_BL_dim_8_l2norm_lrm_256.npy")))
            x_BLCv_wo_first_l = torch.from_numpy(np.load(os.path.join(latent_path, "x_BLCv_wo_first_l_dim_8_l2_norm_lrm_256.npy")))
            if self.text_conditioned:
                text_embedding = torch.from_numpy(np.load(os.path.join(latent_path, "text_embedding_lrm_3dtopia.npy")))
                text_pooler_ouput = torch.from_numpy(np.load(os.path.join(latent_path, "text_pooler_output_lrm_3dtopia.npy")))

                sample.update({
                    'gt_BL': gt_BL,
                    'x_BLCv_wo_first_l': x_BLCv_wo_first_l,
                    'text_embedding': text_embedding,
                    'text_pooler_output': text_pooler_ouput})
            else:
                # Load latent codes and embeddings
                image_embedding = torch.from_numpy(np.load(os.path.join(latent_path, "image_dino_embedding_lrm.npy")))
                image_dino_embedding = image_embedding[1:, :]
                image_dino_pooler_output = image_embedding[0]

                sample.update({
                    'gt_BL': gt_BL,
                    'x_BLCv_wo_first_l': x_BLCv_wo_first_l,
                    'image_dino_pooler_output': image_dino_pooler_output,
                    'image_dino_embedding': image_dino_embedding,
                })


        else:
            raise NotImplementedError(os.path.join(latent_path, "gt_BL_dim_8_l2norm_lrm_256.npy"))

        return sample

    def __len__(self):
        return len(self.chunk_list)

    def __getitem__(self, index):
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
        return sample


def load_data(
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
    infi_sampler=True,
    eval=False,
    **kwargs
):
    dataset_cls = ChunkObjaverseDataset_eval_VAE if eval else ChunkObjaverseDataset_VAE
    collate_fn = chunk_collate_fn

    dataset = dataset_cls(
        file_path,
        reso,
        reso_encoder,
        test=False,
        preprocess=preprocess,
        load_depth=load_depth,
        imgnet_normalize=imgnet_normalize,
        dataset_size=dataset_size,
        **kwargs
    )

    logger.log(f'dataset_cls: {trainer_name}, dataset size: {len(dataset)}')

    if infi_sampler:
        train_sampler = DistributedSampler(dataset=dataset, shuffle=True, drop_last=True)
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
        while True:
            yield from loader

class ChunkObjaverseDataset_VAE(Dataset):
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
            **kwargs):

        super().__init__()
        
        self.read_normal = True
        self.file_path = file_path
        self.chunk_size = 12
        self.gs_cam_format = gs_cam_format

        self.frame_0_as_canonical = frame_0_as_canonical
        self.four_view_for_latent = four_view_for_latent  # export 0 12 30 36, 4 views for reconstruction
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

        # ! load all chunk paths
        self.chunk_list = []
        # ! direclty load from json
        dataset_json = []
        for cl in [  # ! around 17W instances in total. 
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
                cl_dataset_json = json.load(f)[cl]
            dataset_json = dataset_json + cl_dataset_json

        if self.chunk_size == 12:
            self.img_ext = 'png' # ln3diff
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
        # e.g., remove bottom view
        pass

    def read_chunk(self, chunk_path):
        raw_img = imageio.imread(os.path.join(chunk_path, f'raw_img.{self.img_ext}'))
        h, bw, c = raw_img.shape
        raw_img = raw_img.reshape(h, self.chunk_size, -1, c).transpose(
            (1, 0, 2, 3))

        depth_alpha = imageio.imread(
            os.path.join(chunk_path, 'depth_alpha.jpg'))  # 2h 10w
        depth_alpha = depth_alpha.reshape(h * 2, self.chunk_size,
                                          -1).transpose((1, 0, 2))

        depth, alpha = np.split(depth_alpha, 2, axis=1)

        c = np.load(os.path.join(chunk_path, 'c.npy'))

        d_near_far = np.load(os.path.join(chunk_path, 'd_near_far.npy'))
        bbox = np.load(os.path.join(chunk_path, 'bbox.npy'))

        d_near = d_near_far[0].reshape(self.chunk_size, 1, 1)
        d_far = d_near_far[1].reshape(self.chunk_size, 1, 1)
        depth = 1 / ((depth / 255) * (d_far - d_near) + d_near)

        depth[depth > 2.9] = 0.0  # background as 0, follow old tradition

        with open(os.path.join(chunk_path, 'caption_3dtopia.txt'), 'r', encoding="utf-8") as f:
            caption = f.read()

        with open(os.path.join(chunk_path, 'ins.txt'), 'r', encoding="utf-8") as f:
            ins = f.read()

        if self.read_normal:
            normal = imageio.imread(os.path.join(
                chunk_path, 'normal.png')).astype(np.float32) / 255.0

            normal = (normal * 2 - 1).reshape(h, self.chunk_size, -1,
                                              3).transpose((1, 0, 2, 3))
            normal = unity2blender_fix(normal) 
            depth = (depth, normal) 

        return raw_img, depth, c, alpha, bbox, caption, ins

    def __len__(self):
        return len(self.chunk_list)

    def __getitem__(self, index) -> Any:
        sample = self.read_chunk(os.path.join(self.file_path, self.chunk_list[index]))
        sample = self.post_process.paired_post_process_chunk(sample)
        sample = self.post_process.create_dict_nobatch(sample)

        return sample



class ChunkObjaverseDataset_eval_VAE(Dataset):
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
            # frame_0_as_canonical=True,
            frame_0_as_canonical=False,
            split_chunk_size=12,
            mv_input=True,
            append_depth=True,
            pcd_path=None,
            load_pcd=False,
            **kwargs):

        super().__init__()
        # st()
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

        # ! load all chunk paths
        self.chunk_list = []

        dataset_json = []
        for cl in [  # ! around 17W instances in total. 
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
                cl_dataset_json = json.load(f)[cl][0:10]
            dataset_json = dataset_json + cl_dataset_json

    
        if self.chunk_size == 12:
            self.img_ext = 'png' # ln3diff
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


    def read_chunk(self, chunk_path):
        raw_img = imageio.imread(os.path.join(chunk_path, f'raw_img.{self.img_ext}'))
        h, bw, c = raw_img.shape
        raw_img = raw_img.reshape(h, self.chunk_size, -1, c).transpose(
            (1, 0, 2, 3))

        depth_alpha = imageio.imread(
            os.path.join(chunk_path, 'depth_alpha.jpg'))  # 2h 10w
        depth_alpha = depth_alpha.reshape(h * 2, self.chunk_size,
                                          -1).transpose((1, 0, 2))

        depth, alpha = np.split(depth_alpha, 2, axis=1)
        c = np.load(os.path.join(chunk_path, 'c.npy'))

        d_near_far = np.load(os.path.join(chunk_path, 'd_near_far.npy'))
        bbox = np.load(os.path.join(chunk_path, 'bbox.npy'))

        d_near = d_near_far[0].reshape(self.chunk_size, 1, 1)
        d_far = d_near_far[1].reshape(self.chunk_size, 1, 1)
        depth = 1 / ((depth / 255) * (d_far - d_near) + d_near)

        depth[depth > 2.9] = 0.0  # background as 0, follow old tradition

        with open(os.path.join(chunk_path, 'caption_3dtopia.txt'), 'r', encoding="utf-8") as f:
            caption = f.read()

        with open(os.path.join(chunk_path, 'ins.txt'), 'r', encoding="utf-8") as f:
            ins = f.read()
        return raw_img, depth, c, alpha, bbox, caption, ins

    def __len__(self):
        return len(self.chunk_list)

    def __getitem__(self, index) -> Any:
        sample = self.read_chunk(os.path.join(self.file_path, self.chunk_list[index]))
        sample = self.post_process.paired_post_process_chunk(sample)
        sample = self.post_process.create_dict_nobatch(sample)
        return sample


def unity2blender_fix(normal): # up blue, left green, front (towards inside) red
    normal_clone = normal.copy()
    normal_clone[..., 0] = -normal[..., 0] # swap r and g
    normal_clone[..., 1] = -normal[..., 2]
    normal_clone[..., 2] = normal[..., 1]

    return normal_clone