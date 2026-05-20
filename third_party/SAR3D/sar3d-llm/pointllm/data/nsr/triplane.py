# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from threading import local
import torch
import torch.nn as nn
from pointllm.data.utils_var.torch_utils import persistence
from .networks_stylegan2 import Generator as StyleGAN2Backbone
from .networks_stylegan2 import ToRGBLayer, SynthesisNetwork, MappingNetwork
from .volumetric_rendering.renderer import ImportanceRenderer
from .volumetric_rendering.ray_sampler import RaySampler, PatchRaySampler
from pointllm.model import dnnlib
from pdb import set_trace as st
import math

import torch.nn.functional as F
import itertools
from pointllm.model.ldm.modules.diffusionmodules.model import SimpleDecoder, Decoder

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pointllm.model.Instantmesh.camera.perspective_camera import PerspectiveCamera
from pointllm.model.Instantmesh.render.neural_render import NeuralRender
from pointllm.model.Instantmesh.rep_3d.flexicubes_geometry import FlexiCubesGeometry
import numpy as np
import trimesh

def save_obj(pointnp_px3, facenp_fx3, colornp_px3, fpath):

    pointnp_px3 = pointnp_px3 @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    facenp_fx3 = facenp_fx3[:, [2, 1, 0]]

    mesh = trimesh.Trimesh(
        vertices=pointnp_px3, 
        faces=facenp_fx3, 
        vertex_colors=colornp_px3,
    )
    mesh.export(fpath, 'obj')

def sdf_reg_loss_batch(sdf, all_edges):
    sdf_f1x6x2 = sdf[:, all_edges.reshape(-1)].reshape(sdf.shape[0], -1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    # st()
    sdf_diff = F.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()) + \
               F.binary_cross_entropy_with_logits(
                   sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff

def camera_to_spherical_batch(c2ws):
    """
    给定一个 [M, N, 4, 4] 的 Camera to World 矩阵，计算相机位置的极坐标。
    """
    # 提取平移部分 (x, y, z)，即每个 Camera to World 矩阵的最后一列
    positions = c2ws[..., :3, 3]  # 形状 [M, N, 3]

    # x, y, z 分量
    x = positions[..., 0]
    y = positions[..., 1]
    z = positions[..., 2]

    # 计算径向距离 r
    r = torch.sqrt(x**2 + y**2 + z**2)

    # 计算方位角 theta
    theta = torch.atan2(y, x)

    # 计算仰角 phi
    phi = torch.atan2(z, torch.sqrt(x**2 + y**2))

    # 将 theta 和 phi 转换为角度
    theta = torch.rad2deg(theta)  # 方位角转换为度数
    phi = torch.rad2deg(phi)      # 仰角转换为度数

    # 拼接结果为 [M, N, 3] 的张量，分别为 (r, theta, phi)
    spherical_coords = torch.stack((r, theta, phi), dim=-1)

    return spherical_coords

def get_camera_poses(radius=2.5, elevation=30.0, azimuth=180.0):
    # M: number of circular views
    # radius: camera dist to center
    # elevation: elevation degrees of the camera
    # return: (M, 4, 4)
    # assert radius > 0

    elevation = np.deg2rad(elevation)

    camera_positions = []
    # for i in range(M):
    #     azimuth = 2 * np.pi * i / M
    #     x = radius * np.cos(elevation) * np.cos(azimuth)
    #     y = radius * np.cos(elevation) * np.sin(azimuth)
    #     z = radius * np.sin(elevation)
    #     camera_positions.append([x, y, z])
    azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)
    camera_positions.append([x, y, z])
    camera_positions = np.array(camera_positions)
    camera_positions = torch.from_numpy(camera_positions).float()
    # st()
    extrinsics = center_looking_at_camera_pose(camera_positions)
    return extrinsics

def get_camera_poses_batch(radius=2.5, elevation=30.0, azimuth=None):
    """
    计算多个相机的位姿矩阵 (extrinsics).
    :param radius: 相机距离中心的半径
    :param elevation: 相机的仰角（单位：度）
    :param azimuths: 一个批次的方位角列表（单位：度），每个角度对应一个相机位置
    :return: (M, 4, 4) 相机外参矩阵，M 是批次中的相机数量
    """
    
    # assert radius > 0, "Radius must be positive"
    # assert azimuth is not None, "Azimuths cannot be None"
    
    elevation = np.deg2rad(elevation)  # 将仰角转换为弧度
    azimuth = np.deg2rad(azimuth)  # 将方位角列表转换为弧度

    # 计算每个相机的 (x, y, z) 坐标
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)

    # 将相机位置堆叠为 (M, 3) 的形状
    camera_positions = np.stack([x, y, z], axis=-1)  # 形状为 (M, 3)
    camera_positions = torch.from_numpy(camera_positions).float()  # 转为 PyTorch 张量

    # 计算相机外参矩阵
    # st()
    extrinsics = center_looking_at_camera_pose_batch(camera_positions)  # 计算相机外参

    return extrinsics

def center_looking_at_camera_pose_batch(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
    """
    Create OpenGL camera extrinsics from camera locations and look-at positions (supporting batch).

    Args:
        camera_position: (M, 3) 相机位置的张量，M 是 batch 的大小。
        look_at: (M, 3) or (3,) 目标点，默认是原点 (0, 0, 0)，可以是 batch 大小。
        up_world: (M, 3) or (3,) 世界 up 方向，默认是 z 轴方向 (0, 0, 1)，可以是 batch 大小。

    Returns:
        extrinsics: (M, 4, 4) 外参矩阵
    """
    # 默认目标点为原点
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32, device=camera_position.device)

    # 默认世界上方向为 z 轴
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32, device=camera_position.device)

    # 如果 `camera_position` 是 (M, 3)，处理成批量形式
    if camera_position.ndim == 2:  # camera_position 是 (M, 3)
        if look_at.ndim == 1:  # 如果 look_at 是 (3,) 形状的，需要扩展到 (M, 3)
            look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        if up_world.ndim == 1:  # 如果 up_world 是 (3,) 形状的，需要扩展到 (M, 3)
            up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    # 计算 z 轴，表示相机的视线方向（从相机位置指向 look_at）
    z_axis = camera_position - look_at
    z_axis = F.normalize(z_axis, dim=-1).float()

    # 确保 z 轴和 up_world 不平行
    if torch.abs(torch.sum(z_axis * up_world, dim=-1)).max() > 0.999:
        print('Warning: camera up and z-axis are almost parallel')
        up_world = torch.tensor([0, -1, 0], dtype=torch.float32, device=camera_position.device)
        if camera_position.ndim == 2:
            up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    # 计算 x 轴，表示相机的右方向
    x_axis = torch.cross(up_world, z_axis, dim=-1)
    x_axis = F.normalize(x_axis, dim=-1).float()

    # 计算 y 轴，表示相机的上方向
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    y_axis = F.normalize(y_axis, dim=-1).float()

    # 将 x, y, z 轴和相机位置组合为外参矩阵
    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)

    # 将矩阵扩展为 4x4，添加齐次坐标
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)

    return extrinsics

def center_looking_at_camera_pose(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
    """
    Create OpenGL camera extrinsics from camera locations and look-at position.

    camera_position: (M, 3) or (3,)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4) or (3, 4)
    """
    # by default, looking at the origin and world up is z-axis
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    # OpenGL camera: z-backward, x-right, y-up
    z_axis = camera_position - look_at
    z_axis = F.normalize(z_axis, dim=-1).float()
    # st()
    if torch.abs(torch.dot(z_axis[0], up_world[0])) > 0.999:
        print('Warning: camera up and z-axis are almost parallel')
        up_world = torch.tensor([0, -1, 0], dtype=torch.float32).unsqueeze(0)
    # st()
    x_axis = torch.linalg.cross(up_world, z_axis, dim=-1)
    x_axis = F.normalize(x_axis, dim=-1).float()
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    y_axis = F.normalize(y_axis, dim=-1).float()

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)
    return extrinsics

def pad_camera_extrinsics_4x4(extrinsics):
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    extrinsics = torch.cat([extrinsics, padding], dim=-2)
    return extrinsics

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):

    def __init__(
            self,
            z_dim,  # Input latent (Z) dimensionality.
            c_dim,  # Conditioning label (C) dimensionality.
            w_dim,  # Intermediate latent (W) dimensionality.
            img_resolution,  # Output resolution.
            img_channels,  # Number of output color channels.
            sr_num_fp16_res=0,
            mapping_kwargs={},  # Arguments for MappingNetwork.
            rendering_kwargs={},
            sr_kwargs={},
            bcg_synthesis_kwargs={},
            # pifu_kwargs={},
            # ada_kwargs={},  # not used, place holder
            **synthesis_kwargs,  # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.renderer = ImportanceRenderer()
        # if 'PatchRaySampler' in rendering_kwargs:
        #     self.ray_sampler = PatchRaySampler()
        # else:
        #     self.ray_sampler = RaySampler()
        self.backbone = StyleGAN2Backbone(z_dim,
                                          c_dim,
                                          w_dim,
                                          img_resolution=256,
                                          img_channels=32 * 3,
                                          mapping_kwargs=mapping_kwargs,
                                          **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(
            class_name=rendering_kwargs['superresolution_module'],
            channels=32,
            img_resolution=img_resolution,
            sr_num_fp16_res=sr_num_fp16_res,
            sr_antialias=rendering_kwargs['sr_antialias'],
            **sr_kwargs)

        # self.bcg_synthesis = None
        if rendering_kwargs.get('use_background', False):
            self.bcg_synthesis = SynthesisNetwork(
                w_dim,
                img_resolution=self.superresolution.input_resolution,
                img_channels=32,
                **bcg_synthesis_kwargs)
            self.bcg_mapping = MappingNetwork(z_dim=z_dim,
                                              c_dim=c_dim,
                                              w_dim=w_dim,
                                              num_ws=self.num_ws,
                                              **mapping_kwargs)
        # New mapping network for self-adaptive camera pose, dim = 3

        self.decoder = OSGDecoder(
            32, {
                'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                'decoder_output_dim': 32
            })
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs

        self._last_planes = None
        self.pool_256 = torch.nn.AdaptiveAvgPool2d((256, 256))

    def mapping(self,
                z,
                c,
                truncation_psi=1,
                truncation_cutoff=None,
                update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z,
                                     c *
                                     self.rendering_kwargs.get('c_scale', 0),
                                     truncation_psi=truncation_psi,
                                     truncation_cutoff=truncation_cutoff,
                                     update_emas=update_emas)

    def synthesis(self,
                  ws,
                  c,
                  neural_rendering_resolution=None,
                  update_emas=False,
                  cache_backbone=False,
                  use_cached_backbone=False,
                  return_meta=False,
                  return_raw_only=False,
                  **synthesis_kwargs):

        return_sampling_details_flag = self.rendering_kwargs.get(
            'return_sampling_details_flag', False)

        if return_sampling_details_flag:
            return_meta = True

        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        # cam2world_matrix = torch.eye(4, device=c.device).unsqueeze(0).repeat_interleave(c.shape[0], dim=0)
        # c[:, :16] = cam2world_matrix.view(-1, 16)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        H = W = self.neural_rendering_resolution
        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(
            cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(
                ws[:, :self.backbone.num_ws, :],  # ws, BS 14 512
                update_emas=update_emas,
                **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2],
                             planes.shape[-1])  # BS 96 256 256

        # Perform volume rendering
        # st()
        rendering_details = self.renderer(
            planes,
            self.decoder,
            ray_origins,
            ray_directions,
            self.rendering_kwargs,
            #   return_meta=True)
            return_meta=return_meta)

        # calibs = create_calib_matrix(c)
        # all_coords = rendering_details['all_coords']
        # B, num_rays, S, _ = all_coords.shape
        # all_coords_B3N = all_coords.reshape(B, -1, 3).permute(0,2,1)
        # homo_coords = torch.cat([all_coords, torch.zeros_like(all_coords[..., :1])], -1)
        # homo_coords[..., -1] = 1
        # homo_coords = homo_coords.reshape(homo_coords.shape[0], -1, 4)
        # homo_coords = homo_coords.permute(0,2,1)
        # xyz = calibs @ homo_coords
        # xyz = xyz.permute(0,2,1).reshape(B, H, W, S, 4)
        # st()

        # xyz_proj = perspective(all_coords_B3N, calibs)
        # xyz_proj = xyz_proj.permute(0,2,1).reshape(B, H, W, S, 3) # [0,0] - [1,1]
        # st()

        feature_samples, depth_samples, weights_samples = (
            rendering_details[k]
            for k in ['feature_samples', 'depth_samples', 'weights_samples'])

        if return_sampling_details_flag:
            shape_synthesized = rendering_details['shape_synthesized']
        else:
            shape_synthesized = None

        # Reshape into 'raw' neural-rendered image
        feature_image = feature_samples.permute(0, 2, 1).reshape(
            N, feature_samples.shape[-1], H, W).contiguous()  # B 32 H W
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]  # B 3 H W
        if not return_raw_only:
            sr_image = self.superresolution(
                rgb_image,
                feature_image,
                ws[:, -1:, :],  # only use the last layer
                noise_mode=self.rendering_kwargs['superresolution_noise_mode'],
                **{
                    k: synthesis_kwargs[k]
                    for k in synthesis_kwargs.keys() if k != 'noise_mode'
                })
        else:
            sr_image = rgb_image

        ret_dict = {
            'image': sr_image,
            'image_raw': rgb_image,
            'image_depth': depth_image,
            'weights_samples': weights_samples,
            'shape_synthesized': shape_synthesized
        }
        if return_meta:
            ret_dict.update({
                # 'feature_image': feature_image,
                'feature_volume':
                rendering_details['feature_volume'],
                'all_coords':
                rendering_details['all_coords'],
                'weights':
                rendering_details['weights'],
            })

        return ret_dict

    def sample(self,
               coordinates,
               directions,
               z,
               c,
               truncation_psi=1,
               truncation_cutoff=None,
               update_emas=False,
               **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes.
        ws = self.mapping(z,
                          c,
                          truncation_psi=truncation_psi,
                          truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        planes = self.backbone.synthesis(ws,
                                         update_emas=update_emas,
                                         **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2],
                             planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates,
                                       directions, self.rendering_kwargs)

    def sample_mixed(self,
                     coordinates,
                     directions,
                     ws,
                     truncation_psi=1,
                     truncation_cutoff=None,
                     update_emas=False,
                     **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws,
                                         update_emas=update_emas,
                                         **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2],
                             planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates,
                                       directions, self.rendering_kwargs)

    def forward(self,
                z,
                c,
                truncation_psi=1,
                truncation_cutoff=None,
                neural_rendering_resolution=None,
                update_emas=False,
                cache_backbone=False,
                use_cached_backbone=False,
                **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z,
                          c,
                          truncation_psi=truncation_psi,
                          truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        return self.synthesis(
            ws,
            c,
            update_emas=update_emas,
            neural_rendering_resolution=neural_rendering_resolution,
            cache_backbone=cache_backbone,
            use_cached_backbone=use_cached_backbone,
            **synthesis_kwargs)


from .networks_stylegan2 import FullyConnectedLayer

# class OSGDecoder(torch.nn.Module):

#     def __init__(self, n_features, options):
#         super().__init__()
#         self.hidden_dim = 64
#         self.output_dim = options['decoder_output_dim']
#         self.n_features = n_features

#         self.net = torch.nn.Sequential(
#             FullyConnectedLayer(n_features,
#                                 self.hidden_dim,
#                                 lr_multiplier=options['decoder_lr_mul']),
#             torch.nn.Softplus(),
#             FullyConnectedLayer(self.hidden_dim,
#                                 1 + options['decoder_output_dim'],
#                                 lr_multiplier=options['decoder_lr_mul']))

#     def forward(self, sampled_features, ray_directions):
#         # Aggregate features
#         sampled_features = sampled_features.mean(1)
#         x = sampled_features

#         N, M, C = x.shape
#         x = x.view(N * M, C)

#         x = self.net(x)
#         x = x.view(N, M, -1)
#         rgb = torch.sigmoid(x[..., 1:]) * (
#             1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
#         sigma = x[..., 0:1]
#         return {'rgb': rgb, 'sigma': sigma}


@persistence.persistent_class
class OSGDecoder(torch.nn.Module):
    """seperate rgb and sigma"""

    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64
        # self.decoder_output_dim = options['decoder_output_dim']

        # self.net = torch.nn.Sequential(
        #     FullyConnectedLayer(n_features,
        #                         self.hidden_dim,
        #                         lr_multiplier=options['decoder_lr_mul']),
        #     torch.nn.Softplus(),
        #     FullyConnectedLayer(self.hidden_dim,
        #                         1 + options['decoder_output_dim'],
        #                         lr_multiplier=options['decoder_lr_mul']))
        # self.activation = options.get('decoder_activation', 'sigmoid')

        # separate rgb and sigma
        # st()
        self.decoder_output_dim_rgb = 3
        self.decoder_output_dim_sigma = 1

        # rgb
        self.net_rgb = torch.nn.Sequential(
            FullyConnectedLayer(n_features,
                                self.hidden_dim,
                                lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim,
                                3,
                                lr_multiplier=options['decoder_lr_mul']))
        self.activation = options.get('decoder_activation', 'sigmoid')

        # sigma
        self.net_sigma = torch.nn.Sequential(
            FullyConnectedLayer(n_features,
                                self.hidden_dim,
                                lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim,
                                1,
                                lr_multiplier=options['decoder_lr_mul']))

    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N * M, C)

        # x = self.net(x)
        # x = x.view(N, M, -1)
        # rgb = x[..., 1:]
        # sigma = x[..., 0:1]
        x_rgb = self.net_rgb(x)
        x_rgb = x_rgb.view(N, M, -1)
        rgb = x_rgb[..., 0:]
        x_sigma = self.net_sigma(x)
        x_sigma = x_sigma.view(N, M, -1)
        sigma = x_sigma[..., 0:]
        
        if self.activation == "sigmoid":
            # Original EG3D
            rgb = torch.sigmoid(rgb) * (1 + 2 * 0.001) - 0.001
        elif self.activation == "lrelu":
            # StyleGAN2-style, use with toRGB
            rgb = torch.nn.functional.leaky_relu(rgb, 0.2,
                                                 inplace=True) * math.sqrt(2)
        return {'rgb': rgb, 'sigma': sigma}


@persistence.persistent_class
class OSGDecoderInstantMesh(nn.Module):
    """
    Triplane decoder that gives RGB and sigma values from sampled features.
    Using ReLU here instead of Softplus in the original implementation.
    
    Reference:
    EG3D: https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py#L112
    """
    def __init__(self, n_features: int,
                 hidden_dim: int = 64, num_layers: int = 4, activation: nn.Module = nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * n_features, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 1 + 3),
        )
        # init all bias to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, sampled_features, ray_directions):
        # Aggregate features by mean
        # sampled_features = sampled_features.mean(1)
        # Aggregate features by concatenation
        _N, n_planes, _M, _C = sampled_features.shape
        sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(_N, _M, n_planes*_C)
        x = sampled_features

        N, M, C = x.shape
        x = x.contiguous().view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]

        return {'rgb': rgb, 'sigma': sigma}
# @persistence.persistent_class
# class OSGDecoder(torch.nn.Module):

#     def __init__(self, n_features, options):
#         super().__init__()
#         self.hidden_dim = 64
#         self.decoder_output_dim = options['decoder_output_dim']

#         self.net = torch.nn.Sequential(
#             FullyConnectedLayer(n_features,
#                                 self.hidden_dim,
#                                 lr_multiplier=options['decoder_lr_mul']),
#             torch.nn.Softplus(),
#             FullyConnectedLayer(self.hidden_dim,
#                                 1 + options['decoder_output_dim'],
#                                 lr_multiplier=options['decoder_lr_mul']))
#         self.activation = options.get('decoder_activation', 'sigmoid')



#     def forward(self, sampled_features, ray_directions):
#         # Aggregate features
#         sampled_features = sampled_features.mean(1)
#         x = sampled_features

#         N, M, C = x.shape
#         x = x.view(N * M, C)

#         x = self.net(x)
#         x = x.view(N, M, -1)
#         rgb = x[..., 1:]
#         sigma = x[..., 0:1]
#         if self.activation == "sigmoid":
#             # Original EG3D
#             rgb = torch.sigmoid(rgb) * (1 + 2 * 0.001) - 0.001
#         elif self.activation == "lrelu":
#             # StyleGAN2-style, use with toRGB
#             rgb = torch.nn.functional.leaky_relu(rgb, 0.2,
#                                                  inplace=True) * math.sqrt(2)
#         return {'rgb': rgb, 'sigma': sigma}

class LRMOSGDecoder(nn.Module):
    """
    Triplane decoder that gives RGB and sigma values from sampled features.
    Using ReLU here instead of Softplus in the original implementation.
    
    Reference:
    EG3D: https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py#L112
    """
    def __init__(self, n_features: int,
                 hidden_dim: int = 64, num_layers: int = 4, activation: nn.Module = nn.ReLU):
        super().__init__()
        self.decoder_output_dim_rgb = 3
        self.net = nn.Sequential(
            nn.Linear(3 * n_features, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 1 + self.decoder_output_dim_rgb),
        )
        # init all bias to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, sampled_features, ray_directions):
        # Aggregate features by mean
        # sampled_features = sampled_features.mean(1)
        # Aggregate features by concatenation
        _N, n_planes, _M, _C = sampled_features.shape
        sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(_N, _M, n_planes*_C)
        x = sampled_features

        N, M, C = x.shape
        x = x.contiguous().view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]

        return {'rgb': rgb, 'sigma': sigma}


class LRMOSGDecoderMesh(nn.Module):
    """
    Triplane decoder that gives RGB and sigma values from sampled features.
    Using ReLU here instead of Softplus in the original implementation.
    
    Reference:
    EG3D: https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py#L112
    """
    def __init__(self, n_features: int,
                 hidden_dim: int = 64, num_layers: int = 4, activation: nn.Module = nn.ReLU):
        super().__init__()
        # st()
        self.decoder_output_dim_rgb = 3
        self.net_sdf = nn.Sequential(
            nn.Linear(3 * n_features, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 1),
        )
        self.net_rgb = nn.Sequential(
            nn.Linear(3 * n_features, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, self.decoder_output_dim_rgb),
        )
        self.net_deformation = nn.Sequential(
            nn.Linear(3 * n_features, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 3),
        )
        self.net_weight = nn.Sequential(
            nn.Linear(8 * 3 * n_features, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 21),
        )
        # init all bias to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
        
        from nsr.volumetric_rendering.renderer import  generate_planes
        self.plane_axes = generate_planes()

    def get_geometry_prediction(self, planes, sample_coordinates, flexicubes_indices):
        plane_axes = self.plane_axes.to(planes.device)
        # st()
        box_warp = 2.0
        # sampled_features = sample_from_planes(
        #     plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=self.rendering_kwargs['box_warp'])
        sampled_features = sample_from_planes(
            plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=box_warp)
        # st()
        # sdf, deformation, weight = self.decoder.get_geometry_prediction(sampled_features, flexicubes_indices)
        sdf, deformation, weight = self.sdf_get_geometry_prediction(sampled_features, flexicubes_indices)
        return sdf, deformation, weight
    
    def sdf_get_geometry_prediction(self, sampled_features, flexicubes_indices):
        _N, n_planes, _M, _C = sampled_features.shape
        sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(_N, _M, n_planes*_C)

        sdf = self.net_sdf(sampled_features)
        # add torch sigmod value
        sdf = torch.sigmoid(sdf) - 0.5 
        # print('sdf max:', sdf.max(), 'sdf min:', sdf.min())

        deformation = self.net_deformation(sampled_features)

        grid_features = torch.index_select(input=sampled_features, index=flexicubes_indices.reshape(-1), dim=1)
        grid_features = grid_features.reshape(
            sampled_features.shape[0], flexicubes_indices.shape[0], flexicubes_indices.shape[1] * sampled_features.shape[-1])
        weight = self.net_weight(grid_features) * 0.1

        return sdf, deformation, weight
    
    def get_texture_prediction(self, planes, sample_coordinates):
        plane_axes = self.plane_axes.to(planes.device)
        box_warp = 2.0
        # st()
        sampled_features = sample_from_planes(
            plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=box_warp)
            # plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=self.rendering_kwargs['box_warp'])

        # rgb = self.decoder.get_texture_prediction(sampled_features)
        rgb = self.rgb_get_texture_prediction(sampled_features)
        return rgb

    def rgb_get_texture_prediction(self, sampled_features):
        _N, n_planes, _M, _C = sampled_features.shape
        sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(_N, _M, n_planes*_C)

        rgb = self.net_rgb(sampled_features)
        rgb = torch.sigmoid(rgb)*(1 + 2*0.001) - 0.001  # Uses sigmoid clamping from MipNeRF

        return rgb




    # def forward(self, sampled_features, ray_directions):
    #     # Aggregate features by mean
    #     # sampled_features = sampled_features.mean(1)
    #     # Aggregate features by concatenation
    #     _N, n_planes, _M, _C = sampled_features.shape
    #     sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(_N, _M, n_planes*_C)
    #     x = sampled_features

    #     N, M, C = x.shape
    #     x = x.contiguous().view(N*M, C)

    #     x = self.net(x)
    #     x = x.view(N, M, -1)
    #     rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
    #     sigma = x[..., 0:1]

    #     return {'rgb': rgb, 'sigma': sigma}

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """

    # # ORIGINAL
    # N, M, C = coordinates.shape
    # xy_coords = coordinates[..., [0, 1]]
    # xz_coords = coordinates[..., [0, 2]]
    # zx_coords = coordinates[..., [2, 0]]
    # return torch.stack([xy_coords, xz_coords, zx_coords], dim=1).reshape(N*3, M, 2)

    # FIXED
    N, M, _ = coordinates.shape
    xy_coords = coordinates[..., [0, 1]]
    yz_coords = coordinates[..., [1, 2]]
    zx_coords = coordinates[..., [2, 0]]
    return torch.stack([xy_coords, yz_coords, zx_coords],
                       dim=1).reshape(N * 3, M, 2)


def sample_from_planes(plane_axes,
                       plane_features,
                       coordinates,
                       mode='bilinear',
                       padding_mode='zeros',
                       box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    # st()
    # st()
    plane_features = plane_features.view(N * n_planes, C, H, W)
    # plane_features = plane_features.reshape(N * n_planes, C, H, W)

    coordinates = (2 / box_warp) * coordinates  # TODO: add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes,
                                                coordinates).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(
        plane_features,
        projected_coordinates.float(),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features
class Triplane(torch.nn.Module):

    def __init__(
        self,
        c_dim=25,  # Conditioning label (C) dimensionality.
        img_resolution=128,  # Output resolution.
        img_channels=3,  # Number of output color channels.
        out_chans=96,
        triplane_size=224,
        rendering_kwargs={},
        decoder_in_chans=32,
        decoder_output_dim=32,
        sr_num_fp16_res=0,
        sr_kwargs={},
        create_triplane=False, # for overfitting single instance study
        bcg_synthesis_kwargs={},
        lrm_decoder=False,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution  # TODO
        self.img_channels = img_channels
        self.triplane_size = triplane_size

        self.decoder_in_chans = decoder_in_chans
        self.out_chans = out_chans

        self.renderer = ImportanceRenderer()

        if 'PatchRaySampler' in rendering_kwargs:
            self.ray_sampler = PatchRaySampler()
        else:
            self.ray_sampler = RaySampler()
        # st()

        # if lrm_decoder:
        # use lrm decoder here
        if True:
            self.decoder = LRMOSGDecoder(
                decoder_in_chans,)
        else:
            self.decoder = OSGDecoder(
                decoder_in_chans,
                {
                    'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                    # 'decoder_output_dim': 32
                    'decoder_output_dim': decoder_output_dim
                })

        self.neural_rendering_resolution = img_resolution  # TODO
        # self.neural_rendering_resolution = 128  # TODO
        self.rendering_kwargs = rendering_kwargs
        self.create_triplane = create_triplane
        if create_triplane:
            self.planes = nn.Parameter(torch.randn(1, out_chans, 256, 256))

        if bool(sr_kwargs):  # check whether empty
            assert decoder_in_chans == decoder_output_dim, 'tradition'
            if rendering_kwargs['superresolution_module'] in [
                    'utils.torch_utils.components.PixelUnshuffleUpsample',
                    'utils.torch_utils.components.NearestConvSR',
                    'utils.torch_utils.components.NearestConvSR_Residual'
            ]:
                self.superresolution = dnnlib.util.construct_class_by_name(
                    class_name=rendering_kwargs['superresolution_module'],
                    # * for PixelUnshuffleUpsample
                    sr_ratio=2,  # 2x SR, 128 -> 256
                    output_dim=decoder_output_dim,
                    num_out_ch=3,
                )
            else:
                self.superresolution = dnnlib.util.construct_class_by_name(
                    class_name=rendering_kwargs['superresolution_module'],
                    # * for stylegan upsample
                    channels=decoder_output_dim,
                    img_resolution=img_resolution,
                    sr_num_fp16_res=sr_num_fp16_res,
                    sr_antialias=rendering_kwargs['sr_antialias'],
                    **sr_kwargs)
        else:
            self.superresolution = None

        self.bcg_synthesis = None

    # * pure reconstruction
    def forward(
            self,
            planes=None,
            # img,
            c=None,
            ws=None,
            ray_origins=None,
            ray_directions=None,
            z_bcg=None,
            neural_rendering_resolution=None,
            update_emas=False,
            cache_backbone=False,
            use_cached_backbone=False,
            return_meta=False,
            return_raw_only=False,
            sample_ray_only=False,
            fg_bbox=None,
            **synthesis_kwargs):

        # st()
        cam2world_matrix = c[:, :16].reshape(-1, 4, 4)
        # cam2world_matrix = torch.eye(4, device=c.device).unsqueeze(0).repeat_interleave(c.shape[0], dim=0)
        # c[:, :16] = cam2world_matrix.view(-1, 16)
        intrinsics = c[:, 16:25].reshape(-1, 3, 3)
        # st()
        # True here
        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution
        
        # True here
        if ray_directions is None:  # when output video
            H = W = self.neural_rendering_resolution
            # Create a batch of rays for volume rendering
            # ray_origins, ray_directions, ray_bboxes = self.ray_sampler(
            #     cam2world_matrix, intrinsics, neural_rendering_resolution)
            # st()
            # print(cam2world_matrix[0]==cam2world_matrix[1])
            if sample_ray_only: # ! for sampling
                ray_origins, ray_directions, ray_bboxes = self.ray_sampler(
                    cam2world_matrix, intrinsics, 
                    self.rendering_kwargs.get( 'patch_rendering_resolution' ),
                    self.neural_rendering_resolution, fg_bbox)

                # for patch supervision
                ret_dict = {
                    'ray_origins': ray_origins,
                    'ray_directions': ray_directions,
                    'ray_bboxes': ray_bboxes,
                }

                return ret_dict

            else: # ! for rendering
                ray_origins, ray_directions, _ = self.ray_sampler(
                    cam2world_matrix, intrinsics, self.neural_rendering_resolution,
                    self.neural_rendering_resolution)

        else:
            assert ray_origins is not None
            # st()
            H = W = int(ray_directions.shape[1]**
                        0.5)  # dynamically set patch resolution
        # st()
        # ! match the batch size, if not returned
        # False here
        if planes is None:
            assert self.planes is not None
            planes = self.planes.repeat_interleave(c.shape[0], dim=0)
        return_sampling_details_flag = self.rendering_kwargs.get(
            'return_sampling_details_flag', False)

        # True here
        if return_sampling_details_flag:
            return_meta = True

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape

        # Reshape output into three 32-channel planes
        # False here
        if planes.shape[1] == 3 * 2 * self.decoder_in_chans:
            # if isinstance(planes, tuple):
            #     N *= 2
            triplane_bg = True
            # planes = torch.cat(planes, 0) # inference in parallel
            # ray_origins = ray_origins.repeat(2,1,1)
            # ray_directions = ray_directions.repeat(2,1,1)

        else:
            triplane_bg = False

        # assert not triplane_bg

        # ! hard coded, will fix later
        # if planes.shape[1] == 3 * self.decoder_in_chans:
        # else:

        # planes = planes.view(len(planes), 3, self.decoder_in_chans,
        planes = planes.reshape(
            len(planes),
            3,
            -1,  # ! support background plane
            planes.shape[-2],
            planes.shape[-1])  # BS 96 256 256

        # Perform volume rendering
        ImportanceRenderer.forward
        # st()
        if ray_origins.shape[0] == 1:
            rendering_details = self.renderer(planes[0].unsqueeze(0),
                                            self.decoder,
                                            ray_origins,
                                            ray_directions,
                                            self.rendering_kwargs,
                                            return_meta=return_meta)
        else:
            rendering_details = self.renderer(planes,
                                            self.decoder,
                                            ray_origins,
                                            ray_directions,
                                            self.rendering_kwargs,
                                            return_meta=return_meta)       
        # st()
        feature_samples, depth_samples, weights_samples = (
            rendering_details[k]
            for k in ['feature_samples', 'depth_samples', 'weights_samples'])
        
        # True here
        if return_sampling_details_flag:
            shape_synthesized = rendering_details['shape_synthesized']
        else:
            shape_synthesized = None

        # Reshape into 'raw' neural-rendered image
        # represent RGB, depth, mask respectively
        feature_image = feature_samples.permute(0, 2, 1).reshape(
            N, feature_samples.shape[-1], H,
            W).contiguous()  # B 32 H W, in [-1,1]
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        weights_samples = weights_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Generate Background
        # if self.bcg_synthesis:

        #     # bg composition
        #     # if self.decoder.activation == "sigmoid":
        #     #     feature_image = feature_image * 2 - 1 # Scale to (-1, 1), taken from ray marcher

        #     assert isinstance(
        #         z_bcg, torch.Tensor
        #     )  # 512 latents after reparmaterization, reuse the name
        #     # ws_bcg = ws[:,:self.bcg_synthesis.num_ws] if ws_bcg is None else ws_bcg[:,:self.bcg_synthesis.num_ws]

        #     with torch.autocast(device_type='cuda',
        #                         dtype=torch.float16,
        #                         enabled=False):

        #         ws_bcg = self.bcg_mapping(z_bcg, c=None)  # reuse the name
        #         if ws_bcg.size(1) < self.bcg_synthesis.num_ws:
        #             ws_bcg = torch.cat([
        #                 ws_bcg, ws_bcg[:, -1:].repeat(
        #                     1, self.bcg_synthesis.num_ws - ws_bcg.size(1), 1)
        #             ], 1)

        #         bcg_image = self.bcg_synthesis(ws_bcg,
        #                                        update_emas=update_emas,
        #                                        **synthesis_kwargs)
        #     bcg_image = torch.nn.functional.interpolate(
        #         bcg_image,
        #         size=feature_image.shape[2:],
        #         mode='bilinear',
        #         align_corners=False,
        #         antialias=self.rendering_kwargs['sr_antialias'])
        #     feature_image = feature_image + (1 - weights_samples) * bcg_image

        #     # Generate Raw image
        #     assert self.torgb
        #     rgb_image = self.torgb(feature_image,
        #                            ws_bcg[:, -1],
        #                            fused_modconv=False)
        #     rgb_image = rgb_image.to(dtype=torch.float32,
        #                              memory_format=torch.contiguous_format)
        #     # st()
        # else:

        mask_image = weights_samples * (1 + 2 * 0.001) - 0.001
        # False here
        if triplane_bg:
            # true_bs = N // 2
            # weights_samples = weights_samples[:true_bs]
            # mask_image = mask_image[:true_bs]
            # feature_image = feature_image[:true_bs] * mask_image + feature_image[true_bs:] * (1-mask_image) # the first is foreground
            # depth_image = depth_image[:true_bs]

            # ! composited colors
            # rgb_final = (
            #     1 - fg_ret_dict['weights']
            # ) * bg_ret_dict['rgb_final'] + fg_ret_dict[
            #     'feature_samples']  # https://github.com/SizheAn/PanoHead/blob/17ad915941c7e2703d5aa3eb5ff12eac47c90e53/training/triplane.py#L127C45-L127C64

            # ret_dict.update({
            #     'feature_samples': rgb_final,
            # })
            # st()
            feature_image = (1 - mask_image) * rendering_details[
                'bg_ret_dict']['rgb_final'] + feature_image

        rgb_image = feature_image[:, :3]

        # # Run superresolution to get final image
        # False here
        if self.superresolution is not None and not return_raw_only:
            # assert ws is not None, 'feed in [cls] token here for SR module'

            if ws is not None and ws.ndim == 2:
                ws = ws.unsqueeze(
                    1)[:, -1:, :]  # follow stylegan tradition, B, N, C

            sr_image = self.superresolution(
                rgb=rgb_image,
                x=feature_image,
                base_x=rgb_image,
                ws=ws,  # only use the last layer
                noise_mode=self.
                rendering_kwargs['superresolution_noise_mode'],  # none
                **{
                    k: synthesis_kwargs[k]
                    for k in synthesis_kwargs.keys() if k != 'noise_mode'
                })
        else:
            # sr_image = rgb_image
            sr_image = None

        # True here
        if shape_synthesized is not None:
            shape_synthesized.update({
                'image_depth': depth_image,
            })  # for 3D loss easy computation, wrap all 3D in a single dict

        ret_dict = {
            'feature_image': feature_image,
            # 'image_raw': feature_image[:, :3],
            'image_raw': rgb_image,
            'image_depth': depth_image,
            'weights_samples': weights_samples,
            # 'silhouette': mask_image,
            # 'silhouette_normalized_3channel': (mask_image*2-1).repeat_interleave(3,1), # N 3 H W
            'shape_synthesized': shape_synthesized,
            "image_mask": mask_image,
        }

        if sr_image is not None:
            ret_dict.update({
                'image_sr': sr_image,
            })

        if return_meta:
            ret_dict.update({
                'feature_volume':
                rendering_details['feature_volume'],
                'all_coords':
                rendering_details['all_coords'],
                'weights':
                rendering_details['weights'],
            })

        return ret_dict


# TODO: write function for mesh
class TriplaneMesh(torch.nn.Module):

    def __init__(
        self,
        c_dim=25,  # Conditioning label (C) dimensionality.
        img_resolution=128,  # Output resolution.
        img_channels=3,  # Number of output color channels.
        out_chans=96,
        triplane_size=224,
        rendering_kwargs={},
        decoder_in_chans=32,
        decoder_output_dim=32,
        sr_num_fp16_res=0,
        sr_kwargs={},
        create_triplane=False, # for overfitting single instance study
        bcg_synthesis_kwargs={},
        lrm_decoder=False,
        grid_res=64,
        grid_scale=2.1,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution  # TODO
        self.img_channels = img_channels
        self.triplane_size = triplane_size

        self.decoder_in_chans = decoder_in_chans
        self.out_chans = out_chans

        self.renderer = ImportanceRenderer()

        if 'PatchRaySampler' in rendering_kwargs:
            self.ray_sampler = PatchRaySampler()
        else:
            self.ray_sampler = RaySampler()
        # st()
        if lrm_decoder:
            # self.decoder = LRMOSGDecoder(
            #     decoder_in_chans,)
            self.decoder = LRMOSGDecoderMesh(
                decoder_in_chans,)
        else:
            self.decoder = OSGDecoder(
                decoder_in_chans,
                {
                    'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                    # 'decoder_output_dim': 32
                    'decoder_output_dim': decoder_output_dim
                })

        self.neural_rendering_resolution = img_resolution  # TODO
        # self.neural_rendering_resolution = 128  # TODO
        self.rendering_kwargs = rendering_kwargs
        self.create_triplane = create_triplane
        if create_triplane:
            self.planes = nn.Parameter(torch.randn(1, out_chans, 256, 256))

        if bool(sr_kwargs):  # check whether empty
            assert decoder_in_chans == decoder_output_dim, 'tradition'
            if rendering_kwargs['superresolution_module'] in [
                    'utils.torch_utils.components.PixelUnshuffleUpsample',
                    'utils.torch_utils.components.NearestConvSR',
                    'utils.torch_utils.components.NearestConvSR_Residual'
            ]:
                self.superresolution = dnnlib.util.construct_class_by_name(
                    class_name=rendering_kwargs['superresolution_module'],
                    # * for PixelUnshuffleUpsample
                    sr_ratio=2,  # 2x SR, 128 -> 256
                    output_dim=decoder_output_dim,
                    num_out_ch=3,
                )
            else:
                self.superresolution = dnnlib.util.construct_class_by_name(
                    class_name=rendering_kwargs['superresolution_module'],
                    # * for stylegan upsample
                    channels=decoder_output_dim,
                    img_resolution=img_resolution,
                    sr_num_fp16_res=sr_num_fp16_res,
                    sr_antialias=rendering_kwargs['sr_antialias'],
                    **sr_kwargs)
        else:
            self.superresolution = None

        self.bcg_synthesis = None

        # InstantMesh parameters
        self.grid_res = grid_res
        self.grid_scale = grid_scale
        self.deformation_multiplier = 4.0

    def init_flexicubes_geometry(self, device, fovy=50.0):
        # from guided_diffusion import dist_util
        # device = dist_util.dev()
        camera = PerspectiveCamera(fovy=fovy, device=device)
        renderer = NeuralRender(device, camera_model=camera)
        self.geometry = FlexiCubesGeometry(
            grid_res=self.grid_res, 
            scale=self.grid_scale, 
            renderer=renderer, 
            render_type='neural_render',
            device=device,
        )
    
    # * pure reconstruction
    def forward(
            self,
            planes=None,
            # img,
            c=None,
            ws=None,
            ray_origins=None,
            ray_directions=None,
            z_bcg=None,
            neural_rendering_resolution=None,
            update_emas=False,
            cache_backbone=False,
            use_cached_backbone=False,
            return_meta=False,
            return_raw_only=False,
            sample_ray_only=False,
            fg_bbox=None,
            **synthesis_kwargs):

        
        # cam2world_matrix = c[:, :16].reshape(-1, 4, 4)
        if c.shape[0] == 1 and c.shape[1] == 25:
            c = c.unsqueeze(0)
            render_c2ws = c[:,:,:16].reshape(-1, 1, 4, 4)
        else:
            render_c2ws = c[:,:,:16].reshape(-1, 6, 4, 4)
        ori = render_c2ws.clone()
        # render_c2ws = render_c2ws_ori.clone()
        # st()
        transform_R = torch.tensor([[1,0,0], [0,-1,0],[0,0,-1]], dtype=render_c2ws.dtype).to(render_c2ws.device)
        # R_T = torch.tensor([[1,0,0],[0,1,0], [0,0,1]], dtype=render_c2ws.dtype).to(render_c2ws.device)
        render_c2ws[:,:,:3,:3] = torch.matmul(render_c2ws[:,:,:3,:3], transform_R.unsqueeze(0).unsqueeze(0))
        render_c2ws[:,:,:3,3] = 2.0 * render_c2ws[:,:,:3,3]
        # spherical_coordinates = camera_to_spherical_batch(render_c2ws).cpu().numpy()
        # for i in range(spherical_coordinates.shape[0]):
        #     for j in range(spherical_coordinates.shape[1]):
        #         render_c2ws[i,j,:3,:3] = torch.matmul(render_c2ws[i,j,:3,:3], transform_R)
        #         render_c2ws[i,j,:3,3] = 2.0 * render_c2ws[i,j,:3,3]
                # if j != 0:
                #     render_c2ws[i,j] = get_camera_poses(radius=spherical_coordinates[i,j,0], azimuth=spherical_coordinates[i,j,1], elevation=spherical_coordinates[i,j,2])
                # else:
                #     render_c2ws[i,j,:3,:3] = torch.matmul(transform_R, render_c2ws[i,j,:3,:3])
                #     render_c2ws[i,j,:3,3] = torch.matmul(R_T, render_c2ws[i,j,:3,3].unsqueeze(1)).squeeze(1)
        # st()
        # look down
        # pose = get_camera_poses(radius=1.75, azimuth=0, elevation=89.9)
        # transform_R = torch.tensor([[1,0,0], [0,-1,0],[0,0,-1]], dtype=render_c2ws.dtype).to(render_c2ws.device)
        # render_c2ws[:,:,:3,:3] = torch.matmul(transform_R, render_c2ws_ori[:,:,:3,:3])
        # R_T = torch.tensor([[0,1,0],[1,0,0], [0,0,1]], dtype=render_c2ws.dtype).to(render_c2ws.device)
        # render_c2ws[:,:,0:3,3] = torch.matmul(R_T, render_c2ws_ori[:,:,0:3,3].unsqueeze(3)).squeeze(3)
        # Rotation = torch.tensor([[1,0,0,0],[0,1,0,0], [0,0,1,0], [0,0,0,1]], dtype=render_c2ws.dtype).to(render_c2ws.device)
        # Rotation = torch.tensor([[1,0,0,0],[0,0,-1,0], [0,1,0,0], [0,0,0,1]], dtype=render_c2ws.dtype).to(render_c2ws.device)
        # Rotation = torch.tensor([[1,0,0,0],[0,-1,0,0], [0,1,0,0], [0,0,0,1]], dtype=render_c2ws.dtype).to(render_c2ws.device)
        # Rotation = torch.tensor([[1,0,0,0],[0,0,1,0], [0,-1,0,0], [0,0,0,1]], dtype=render_c2ws.dtype).to(render_c2ws.device)
        # Rotation = torch.tensor([[0,0,1,0],[0,1,0,0], [-1,0,0,0], [0,0,0,1]], dtype=render_c2ws.dtype).to(render_c2ws.device)
        # render_c2ws = torch.matmul(Rotation, render_c2ws)
        # render_c2ws = pose.repeat(c.shape[0], c.shape[1], 1, 1).to(c.device)
        render_w2cs = torch.linalg.inv(render_c2ws.float())
        # st()
        planes = planes.reshape(
            len(planes),
            3,
            -1,  # ! support background plane
            planes.shape[-2],
            planes.shape[-1])  # BS 96 256 256

        out = self.forward_geometry(
            planes = planes,
            render_cameras = render_w2cs,
            render_size = self.neural_rendering_resolution
        )
        # st()
        # export mesh
        # with torch.no_grad():
        #     mesh_out = self.extract_mesh(planes[0].unsqueeze(0).float(), use_texture_map=False)
        #     vertices, faces, vertex_colors = mesh_out
        #     save_obj(vertices, faces, vertex_colors, "/mnt/slurm_home/ywchen/projects/LN3Diff_Ori/LN3diff_VAR/mesh_tex_0.obj")

        # st()

        return out
        # # cam2world_matrix = torch.eye(4, device=c.device).unsqueeze(0).repeat_interleave(c.shape[0], dim=0)
        # # c[:, :16] = cam2world_matrix.view(-1, 16)
        # # intrinsics = c[:,:, 16:25].reshape(-1, 6, 3, 3)
        # # intrinsics = c[:, 16:25].reshape(-1, 3, 3)
        # # st()
        # # True here
        # if neural_rendering_resolution is None:
        #     neural_rendering_resolution = self.neural_rendering_resolution
        # else:
        #     self.neural_rendering_resolution = neural_rendering_resolution
        
        # H = W = self.neural_rendering_resolution
        # # True here
        # # if ray_directions is None:  # when output video
        # #     H = W = self.neural_rendering_resolution
        # #     # Create a batch of rays for volume rendering
        # #     # ray_origins, ray_directions, ray_bboxes = self.ray_sampler(
        # #     #     cam2world_matrix, intrinsics, neural_rendering_resolution)
        # #     # st()
        # #     # print(cam2world_matrix[0]==cam2world_matrix[1])
        # #     if sample_ray_only: # ! for sampling
        # #         ray_origins, ray_directions, ray_bboxes = self.ray_sampler(
        # #             cam2world_matrix, intrinsics, 
        # #             self.rendering_kwargs.get( 'patch_rendering_resolution' ),
        # #             self.neural_rendering_resolution, fg_bbox)

        # #         # for patch supervision
        # #         ret_dict = {
        # #             'ray_origins': ray_origins,
        # #             'ray_directions': ray_directions,
        # #             'ray_bboxes': ray_bboxes,
        # #         }

        # #         return ret_dict

        # #     else: # ! for rendering
        # #         ray_origins, ray_directions, _ = self.ray_sampler(
        # #             cam2world_matrix, intrinsics, self.neural_rendering_resolution,
        # #             self.neural_rendering_resolution)

        # # else:
        # #     assert ray_origins is not None
        # #     # st()
        # #     H = W = int(ray_directions.shape[1]**
        # #                 0.5)  # dynamically set patch resolution
        # # st()
        # # ! match the batch size, if not returned
        # # False here
        # # if planes is None:
        # #     assert self.planes is not None
        # #     planes = self.planes.repeat_interleave(c.shape[0], dim=0)
        # return_sampling_details_flag = self.rendering_kwargs.get(
        #     'return_sampling_details_flag', False)

        # # True here
        # if return_sampling_details_flag:
        #     return_meta = True

        # # Create triplanes by running StyleGAN backbone
        # N, M, _ = ray_origins.shape

        # # Reshape output into three 32-channel planes
        # # False here
        # # if planes.shape[1] == 3 * 2 * self.decoder_in_chans:
        # #     # if isinstance(planes, tuple):
        # #     #     N *= 2
        # #     triplane_bg = True
        # #     # planes = torch.cat(planes, 0) # inference in parallel
        # #     # ray_origins = ray_origins.repeat(2,1,1)
        # #     # ray_directions = ray_directions.repeat(2,1,1)

        # # else:
        # #     triplane_bg = False

        # # assert not triplane_bg

        # # ! hard coded, will fix later
        # # if planes.shape[1] == 3 * self.decoder_in_chans:
        # # else:

        # # planes = planes.view(len(planes), 3, self.decoder_in_chans,
        # planes = planes.reshape(
        #     len(planes),
        #     3,
        #     -1,  # ! support background plane
        #     planes.shape[-2],
        #     planes.shape[-1])  # BS 96 256 256

        # # Perform volume rendering
        # ImportanceRenderer.forward
        # # st()
        # if ray_origins.shape[0] == 1:
        #     rendering_details = self.renderer(planes[0].unsqueeze(0),
        #                                     self.decoder,
        #                                     ray_origins,
        #                                     ray_directions,
        #                                     self.rendering_kwargs,
        #                                     return_meta=return_meta)
        # else:
        #     rendering_details = self.renderer(planes,
        #                                     self.decoder,
        #                                     ray_origins,
        #                                     ray_directions,
        #                                     self.rendering_kwargs,
        #                                     return_meta=return_meta)       
        # # st()
        # feature_samples, depth_samples, weights_samples = (
        #     rendering_details[k]
        #     for k in ['feature_samples', 'depth_samples', 'weights_samples'])
        
        # # True here
        # if return_sampling_details_flag:
        #     shape_synthesized = rendering_details['shape_synthesized']
        # else:
        #     shape_synthesized = None

        # # Reshape into 'raw' neural-rendered image
        # # represent RGB, depth, mask respectively
        # feature_image = feature_samples.permute(0, 2, 1).reshape(
        #     N, feature_samples.shape[-1], H,
        #     W).contiguous()  # B 32 H W, in [-1,1]
        # depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        # weights_samples = weights_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # # Generate Background
        # # if self.bcg_synthesis:

        # #     # bg composition
        # #     # if self.decoder.activation == "sigmoid":
        # #     #     feature_image = feature_image * 2 - 1 # Scale to (-1, 1), taken from ray marcher

        # #     assert isinstance(
        # #         z_bcg, torch.Tensor
        # #     )  # 512 latents after reparmaterization, reuse the name
        # #     # ws_bcg = ws[:,:self.bcg_synthesis.num_ws] if ws_bcg is None else ws_bcg[:,:self.bcg_synthesis.num_ws]

        # #     with torch.autocast(device_type='cuda',
        # #                         dtype=torch.float16,
        # #                         enabled=False):

        # #         ws_bcg = self.bcg_mapping(z_bcg, c=None)  # reuse the name
        # #         if ws_bcg.size(1) < self.bcg_synthesis.num_ws:
        # #             ws_bcg = torch.cat([
        # #                 ws_bcg, ws_bcg[:, -1:].repeat(
        # #                     1, self.bcg_synthesis.num_ws - ws_bcg.size(1), 1)
        # #             ], 1)

        # #         bcg_image = self.bcg_synthesis(ws_bcg,
        # #                                        update_emas=update_emas,
        # #                                        **synthesis_kwargs)
        # #     bcg_image = torch.nn.functional.interpolate(
        # #         bcg_image,
        # #         size=feature_image.shape[2:],
        # #         mode='bilinear',
        # #         align_corners=False,
        # #         antialias=self.rendering_kwargs['sr_antialias'])
        # #     feature_image = feature_image + (1 - weights_samples) * bcg_image

        # #     # Generate Raw image
        # #     assert self.torgb
        # #     rgb_image = self.torgb(feature_image,
        # #                            ws_bcg[:, -1],
        # #                            fused_modconv=False)
        # #     rgb_image = rgb_image.to(dtype=torch.float32,
        # #                              memory_format=torch.contiguous_format)
        # #     # st()
        # # else:

        # mask_image = weights_samples * (1 + 2 * 0.001) - 0.001
        # # False here
        # if triplane_bg:
        #     # true_bs = N // 2
        #     # weights_samples = weights_samples[:true_bs]
        #     # mask_image = mask_image[:true_bs]
        #     # feature_image = feature_image[:true_bs] * mask_image + feature_image[true_bs:] * (1-mask_image) # the first is foreground
        #     # depth_image = depth_image[:true_bs]

        #     # ! composited colors
        #     # rgb_final = (
        #     #     1 - fg_ret_dict['weights']
        #     # ) * bg_ret_dict['rgb_final'] + fg_ret_dict[
        #     #     'feature_samples']  # https://github.com/SizheAn/PanoHead/blob/17ad915941c7e2703d5aa3eb5ff12eac47c90e53/training/triplane.py#L127C45-L127C64

        #     # ret_dict.update({
        #     #     'feature_samples': rgb_final,
        #     # })
        #     # st()
        #     feature_image = (1 - mask_image) * rendering_details[
        #         'bg_ret_dict']['rgb_final'] + feature_image

        # rgb_image = feature_image[:, :3]

        # # # Run superresolution to get final image
        # # False here
        # if self.superresolution is not None and not return_raw_only:
        #     # assert ws is not None, 'feed in [cls] token here for SR module'

        #     if ws is not None and ws.ndim == 2:
        #         ws = ws.unsqueeze(
        #             1)[:, -1:, :]  # follow stylegan tradition, B, N, C

        #     sr_image = self.superresolution(
        #         rgb=rgb_image,
        #         x=feature_image,
        #         base_x=rgb_image,
        #         ws=ws,  # only use the last layer
        #         noise_mode=self.
        #         rendering_kwargs['superresolution_noise_mode'],  # none
        #         **{
        #             k: synthesis_kwargs[k]
        #             for k in synthesis_kwargs.keys() if k != 'noise_mode'
        #         })
        # else:
        #     # sr_image = rgb_image
        #     sr_image = None

        # # True here
        # if shape_synthesized is not None:
        #     shape_synthesized.update({
        #         'image_depth': depth_image,
        #     })  # for 3D loss easy computation, wrap all 3D in a single dict
        # # st()
        # ret_dict = {
        #     'feature_image': feature_image,
        #     # 'image_raw': feature_image[:, :3],
        #     'image_raw': rgb_image,
        #     'image_depth': depth_image,
        #     'weights_samples': weights_samples,
        #     # 'silhouette': mask_image,
        #     # 'silhouette_normalized_3channel': (mask_image*2-1).repeat_interleave(3,1), # N 3 H W
        #     'shape_synthesized': shape_synthesized,
        #     "image_mask": mask_image,
        # }

        # if sr_image is not None:
        #     ret_dict.update({
        #         'image_sr': sr_image,
        #     })

        # if return_meta:
        #     ret_dict.update({
        #         'feature_volume':
        #         rendering_details['feature_volume'],
        #         'all_coords':
        #         rendering_details['all_coords'],
        #         'weights':
        #         rendering_details['weights'],
        #     })

        # return ret_dict
    
    def forward_geometry(self, planes, render_cameras, render_size=256):
        '''
        Main function of our Generator. It first generate 3D mesh, then render it into 2D image
        with given `render_cameras`.
        :param planes: triplane features
        :param render_cameras: cameras to render generated 3D shape
        '''
        # st()
        # here render_cameras are world to camera matrix
        B, NV = render_cameras.shape[:2]

        # Generate 3D mesh first
        # var test
        if planes.shape[0] == 12:
            planes = planes[0].unsqueeze(0)
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(planes)
        # st()

        # save_mesh
        # import trimesh
        # mesh = trimesh.Trimesh(vertices=mesh_v[3].float().detach().cpu(), faces=mesh_f[3].float().detach().cpu())
        # mesh.export('/mnt/slurm_home/ywchen/projects/LN3Diff_Ori/LN3diff_VAR/new_mesh.obj')
        # var test
        if planes.shape[0] == 1:
            pass
        else:
            mesh_v = mesh_v + mesh_v
            mesh_f = mesh_f + mesh_f

        # Render the mesh into 2D image (get 3d position of each image plane)
        cam_mv = render_cameras
        run_n_view = cam_mv.shape[1]
        # st()
        antilias_mask, hard_mask, tex_pos, depth, normal = self.render_mesh(mesh_v, mesh_f, cam_mv, render_size=render_size)
        # st()
        tex_hard_mask = hard_mask
        # st()
        tex_pos = [torch.cat([pos[i_view:i_view + 1] for i_view in range(run_n_view)], dim=2) for pos in tex_pos]
        # st()
        tex_hard_mask = torch.cat(
            [torch.cat(
                [tex_hard_mask[i * run_n_view + i_view: i * run_n_view + i_view + 1]
                 for i_view in range(run_n_view)], dim=2)
                # for i in range(planes.shape[0])], dim=0)
                for i in range(planes.shape[0] * 2)], dim=0)

        # Querying the texture field to predict the texture feature for each pixel on the image
        # st()
        # repeat_planes = planes.repeat(2,1,1,1,1)
        # tex_feat = self.get_texture_prediction(planes, tex_pos, tex_hard_mask)
        # st()
        # var test
        if planes.shape[0] == 1:
            tex_feat = self.get_texture_prediction(planes, tex_pos, tex_hard_mask)
        else:
            tex_feat = self.get_texture_prediction(planes.repeat(2,1,1,1,1), tex_pos, tex_hard_mask)
        # st()
        background_feature = torch.ones_like(tex_feat)      # white background

        # Merge them together
        img_feat = tex_feat * tex_hard_mask + background_feature * (1 - tex_hard_mask)
        # st()

        # We should split it back to the original image shape
        img_feat = torch.cat(
            [torch.cat(
                [img_feat[i:i + 1, :, render_size * i_view: render_size * (i_view + 1)]
                 for i_view in range(run_n_view)], dim=0) for i in range(len(tex_pos))], dim=0)
        # st()
        img = img_feat.clamp(0, 1).permute(0, 3, 1, 2)
        # put it in [-1,1]
        img = img * 2 - 1
        antilias_mask = antilias_mask.permute(0, 3, 1, 2)
        depth = -depth.permute(0, 3, 1, 2)
        normal = normal * 2 - 1
        # normal_mask = (normal == torch.tensor([-1., -1., -1.]).to(normal.device).to(normal.dtype)).all(dim=-1)
        # normal[normal_mask] = torch.tensor([-0.0039, -1.0000, 0.0039]).to(normal.device).to(normal.dtype)
        normal = normal.permute(0, 3, 1, 2)

        # st()
        # img = img_feat.clamp(0, 1).permute(0, 3, 1, 2).unflatten(0, (B, NV))
        # antilias_mask = antilias_mask.permute(0, 3, 1, 2).unflatten(0, (B, NV))
        # depth = -depth.permute(0, 3, 1, 2).unflatten(0, (B, NV))        # transform negative depth to positive
        # normal = normal.permute(0, 3, 1, 2).unflatten(0, (B, NV))

        # calculate the sdf regularization loss
        sdf_reg_loss_entropy = sdf_reg_loss_batch(sdf, self.geometry.all_edges).mean() * 0.01
        _, flexicubes_surface_reg, flexicubes_weight_reg = sdf_reg_loss
        flexicubes_surface_reg = flexicubes_surface_reg.mean() * 0.5
        flexicubes_weight_reg = flexicubes_weight_reg.mean() * 0.1
        # loss_reg = sdf_reg_loss_entropy + flexicubes_surface_reg + flexicubes_weight_reg

        out = {
            # 'img': img,
            'image_raw': img,
            # 'mask': antilias_mask,
            'image_mask': antilias_mask,
            # 'depth': depth,
            'image_depth_mesh': depth,
            'image_normal_mesh': normal,
            # 'sdf': sdf,
            # 'mesh_v': mesh_v,
            # 'mesh_f': mesh_f,
            # 'sdf_reg_loss': sdf_reg_loss,
            # 'calculated_sdf_reg_loss': loss_reg,
            'sdf_reg_loss_entropy': sdf_reg_loss_entropy,
            'flexicubes_surface_reg': flexicubes_surface_reg,
            'flexicubes_weight_reg': flexicubes_weight_reg,
        }
        return out
    
    def render_mesh(self, mesh_v, mesh_f, cam_mv, render_size=256):
        '''
        Function to render a generated mesh with nvdiffrast
        :param mesh_v: List of vertices for the mesh
        :param mesh_f: List of faces for the mesh
        :param cam_mv:  4x4 rotation matrix
        :return:
        '''
        return_value_list = []
        # st()
        for i_mesh in range(len(mesh_v)):
            return_value = self.geometry.render_mesh(
                mesh_v[i_mesh],
                mesh_f[i_mesh].int(),
                cam_mv[i_mesh],
                resolution=render_size,
                hierarchical_mask=False
            )
            return_value_list.append(return_value)
        # st()
        return_keys = return_value_list[0].keys()
        return_value = dict()
        for k in return_keys:
            value = [v[k] for v in return_value_list]
            return_value[k] = value

        mask = torch.cat(return_value['mask'], dim=0)
        hard_mask = torch.cat(return_value['hard_mask'], dim=0)
        tex_pos = return_value['tex_pos']
        depth = torch.cat(return_value['depth'], dim=0)
        normal = torch.cat(return_value['normal'], dim=0)
        return mask, hard_mask, tex_pos, depth, normal
    
    def get_texture_prediction(self, planes, tex_pos, hard_mask=None):
        '''
        Predict Texture given triplanes
        :param planes: the triplane feature map
        :param tex_pos: Position we want to query the texture field
        :param hard_mask: 2D silhoueete of the rendered image
        '''
        tex_pos = torch.cat(tex_pos, dim=0)
        if not hard_mask is None:
            tex_pos = tex_pos * hard_mask.float()
        batch_size = tex_pos.shape[0]
        tex_pos = tex_pos.reshape(batch_size, -1, 3)
        ###################
        # We use mask to get the texture location (to save the memory)
        if hard_mask is not None:
            n_point_list = torch.sum(hard_mask.long().reshape(hard_mask.shape[0], -1), dim=-1)
            sample_tex_pose_list = []
            max_point = n_point_list.max()
            expanded_hard_mask = hard_mask.reshape(batch_size, -1, 1).expand(-1, -1, 3) > 0.5
            for i in range(tex_pos.shape[0]):
                tex_pos_one_shape = tex_pos[i][expanded_hard_mask[i]].reshape(1, -1, 3)
                if tex_pos_one_shape.shape[1] < max_point:
                    tex_pos_one_shape = torch.cat(
                        [tex_pos_one_shape, torch.zeros(
                            1, max_point - tex_pos_one_shape.shape[1], 3,
                            device=tex_pos_one_shape.device, dtype=torch.float32)], dim=1)
                sample_tex_pose_list.append(tex_pos_one_shape)
            tex_pos = torch.cat(sample_tex_pose_list, dim=0)
        # st()
        tex_feat = torch.utils.checkpoint.checkpoint(
            # self.synthesizer.get_texture_prediction,
            self.decoder.get_texture_prediction,
            planes, 
            tex_pos,
            use_reentrant=False,
        )

        if hard_mask is not None:
            # final_tex_feat = torch.zeros(
            #     planes.shape[0], hard_mask.shape[1] * hard_mask.shape[2], tex_feat.shape[-1], device=tex_feat.device)
            final_tex_feat = torch.zeros(
                planes.shape[0], hard_mask.shape[1] * hard_mask.shape[2], tex_feat.shape[-1], device=tex_feat.device).to(tex_feat.dtype)
            # st()
            expanded_hard_mask = hard_mask.reshape(hard_mask.shape[0], -1, 1).expand(-1, -1, final_tex_feat.shape[-1]) > 0.5
            for i in range(planes.shape[0]):
                # st()
                final_tex_feat[i][expanded_hard_mask[i]] = tex_feat[i][:n_point_list[i]].reshape(-1)
            tex_feat = final_tex_feat

        return tex_feat.reshape(planes.shape[0], hard_mask.shape[1], hard_mask.shape[2], tex_feat.shape[-1])
    
    def extract_mesh(
        self, 
        planes: torch.Tensor, 
        use_texture_map: bool = False,
        texture_resolution: int = 1024,
        **kwargs,
    ):
        '''
        Extract a 3D mesh from FlexiCubes. Only support batch_size 1.
        :param planes: triplane features
        :param use_texture_map: use texture map or vertex color
        :param texture_resolution: the resolution of texure map
        '''
        assert planes.shape[0] == 1
        device = planes.device

        # predict geometry first
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(planes)
        vertices, faces = mesh_v[0], mesh_f[0]

        if not use_texture_map:
            # query vertex colors
            vertices_tensor = vertices.unsqueeze(0)
            # vertices_colors = self.synthesizer.get_texture_prediction(
            # st()
            vertices_colors = self.decoder.get_texture_prediction(
                # planes, vertices_tensor).clamp(0, 1).squeeze(0).cpu().numpy()
                planes, vertices_tensor.float()).float().clamp(0, 1).squeeze(0).cpu().numpy()
            # vertices_colors = (vertices_colors * 255).astype(np.uint8)
            vertices_colors = (vertices_colors * 255).astype(np.uint8)

            # return vertices.cpu().numpy(), faces.cpu().numpy(), vertices_colors
            return vertices.float().cpu().numpy(), faces.cpu().numpy(), vertices_colors

        # use x-atlas to get uv mapping for the mesh
        import nvdiffrast.torch as dr
        from utils.mesh_util import xatlas_uvmap
        ctx = dr.RasterizeCudaContext(device=device)
        uvs, mesh_tex_idx, gb_pos, tex_hard_mask = xatlas_uvmap(
            self.geometry.renderer.ctx, vertices, faces, resolution=texture_resolution)
        tex_hard_mask = tex_hard_mask.float()

        # query the texture field to get the RGB color for texture map
        tex_feat = self.get_texture_prediction(
            planes, [gb_pos], tex_hard_mask)
        background_feature = torch.zeros_like(tex_feat)
        img_feat = torch.lerp(background_feature, tex_feat, tex_hard_mask)
        texture_map = img_feat.permute(0, 3, 1, 2).squeeze(0)

        return vertices, faces, uvs, mesh_tex_idx, texture_map
    
    def get_geometry_prediction(self, planes=None):
        '''
        Function to generate mesh with give triplanes
        :param planes: triplane features
        '''
        # Step 1: first get the sdf and deformation value for each vertices in the tetrahedon grid.
        sdf, deformation, sdf_reg_loss, weight = self.get_sdf_deformation_prediction(planes)
        # st()
        # v_deformed = self.geometry.verts.unsqueeze(dim=0).expand(sdf.shape[0], -1, -1) + deformation
        v_deformed = self.geometry.verts.to(deformation.dtype).unsqueeze(dim=0).expand(sdf.shape[0], -1, -1) + deformation
        tets = self.geometry.indices
        n_batch = planes.shape[0]
        v_list = []
        f_list = []
        flexicubes_surface_reg_list = []
        # st()
        # Step 2: Using marching tet to obtain the mesh
        for i_batch in range(n_batch):
            verts, faces, flexicubes_surface_reg = self.geometry.get_mesh(
                v_deformed[i_batch], 
                sdf[i_batch].squeeze(dim=-1),
                with_uv=False, 
                indices=tets, 
                weight_n=weight[i_batch].squeeze(dim=-1),
                is_training=self.training,
            )
            flexicubes_surface_reg_list.append(flexicubes_surface_reg)
            v_list.append(verts)
            f_list.append(faces)
        
        flexicubes_surface_reg = torch.cat(flexicubes_surface_reg_list).mean()
        flexicubes_weight_reg = (weight ** 2).mean()
        
        return v_list, f_list, sdf, deformation, v_deformed, (sdf_reg_loss, flexicubes_surface_reg, flexicubes_weight_reg)

    def get_sdf_deformation_prediction(self, planes):
        '''
        Predict SDF and deformation for tetrahedron vertices
        :param planes: triplane feature map for the geometry
        '''
        init_position = self.geometry.verts.unsqueeze(0).expand(planes.shape[0], -1, -1)
        
        # Step 1: predict the SDF and deformation
        sdf, deformation, weight = torch.utils.checkpoint.checkpoint(
            # self.synthesizer.get_geometry_prediction,
            self.decoder.get_geometry_prediction,
            planes, 
            init_position, 
            self.geometry.indices,
            use_reentrant=False,
        )

        # Step 2: Normalize the deformation to avoid the flipped triangles.
        deformation = 1.0 / (self.grid_res * self.deformation_multiplier) * torch.tanh(deformation)
        sdf_reg_loss = torch.zeros(sdf.shape[0], device=sdf.device, dtype=torch.float32)

        ####
        # Step 3: Fix some sdf if we observe empty shape (full positive or full negative)
        sdf_bxnxnxn = sdf.reshape((sdf.shape[0], self.grid_res + 1, self.grid_res + 1, self.grid_res + 1))
        sdf_less_boundary = sdf_bxnxnxn[:, 1:-1, 1:-1, 1:-1].reshape(sdf.shape[0], -1)
        pos_shape = torch.sum((sdf_less_boundary > 0).int(), dim=-1)
        neg_shape = torch.sum((sdf_less_boundary < 0).int(), dim=-1)
        zero_surface = torch.bitwise_or(pos_shape == 0, neg_shape == 0)
        # st()
        if torch.sum(zero_surface).item() > 0:
            update_sdf = torch.zeros_like(sdf[0:1])
            max_sdf = sdf.max()
            min_sdf = sdf.min()
            update_sdf[:, self.geometry.center_indices] += (1.0 - min_sdf)  # greater than zero
            update_sdf[:, self.geometry.boundary_indices] += (-1 - max_sdf)  # smaller than zero
            new_sdf = torch.zeros_like(sdf)
            for i_batch in range(zero_surface.shape[0]):
                if zero_surface[i_batch]:
                    new_sdf[i_batch:i_batch + 1] += update_sdf
            update_mask = (new_sdf == 0).float()
            # Regulraization here is used to push the sdf to be a different sign (make it not fully positive or fully negative)
            # st()
            sdf_reg_loss = torch.abs(sdf).mean(dim=-1).mean(dim=-1)
            sdf_reg_loss = sdf_reg_loss * zero_surface.float()
            sdf = sdf * update_mask + new_sdf * (1 - update_mask)

        # Step 4: Here we remove the gradient for the bad sdf (full positive or full negative)
        final_sdf = []
        final_def = []
        for i_batch in range(zero_surface.shape[0]):
            if zero_surface[i_batch]:
                final_sdf.append(sdf[i_batch: i_batch + 1].detach())
                final_def.append(deformation[i_batch: i_batch + 1].detach())
            else:
                final_sdf.append(sdf[i_batch: i_batch + 1])
                final_def.append(deformation[i_batch: i_batch + 1])
        sdf = torch.cat(final_sdf, dim=0)
        deformation = torch.cat(final_def, dim=0)
        return sdf, deformation, sdf_reg_loss, weight
    
class Triplane_fg_bg_plane(Triplane):
    # a separate background plane

    def __init__(self,
                 c_dim=25,
                 img_resolution=128,
                 img_channels=3,
                 out_chans=96,
                 triplane_size=224,
                 rendering_kwargs={},
                 decoder_in_chans=32,
                 decoder_output_dim=32,
                 sr_num_fp16_res=0,
                 sr_kwargs={},
                 bcg_synthesis_kwargs={}):
        super().__init__(c_dim, img_resolution, img_channels, out_chans,
                         triplane_size, rendering_kwargs, decoder_in_chans,
                         decoder_output_dim, sr_num_fp16_res, sr_kwargs,
                         bcg_synthesis_kwargs)

        self.bcg_decoder = Decoder(
            ch=64,  # half channel size
            out_ch=32,
            # ch_mult=(1, 2, 4),
            ch_mult=(1, 2),  # use res=64 for now
            num_res_blocks=2,
            dropout=0.0,
            attn_resolutions=(),
            z_channels=4,
            resolution=64,
            in_channels=3,
        )

    # * pure reconstruction
    def forward(
            self,
            planes,
            bg_plane,
            # img,
            c,
            ws=None,
            z_bcg=None,
            neural_rendering_resolution=None,
            update_emas=False,
            cache_backbone=False,
            use_cached_backbone=False,
            return_meta=False,
            return_raw_only=False,
            **synthesis_kwargs):

        # ! match the batch size
        if planes is None:
            assert self.planes is not None
            planes = self.planes.repeat_interleave(c.shape[0], dim=0)
        return_sampling_details_flag = self.rendering_kwargs.get(
            'return_sampling_details_flag', False)

        if return_sampling_details_flag:
            return_meta = True

        cam2world_matrix = c[:, :16].reshape(-1, 4, 4)
        # cam2world_matrix = torch.eye(4, device=c.device).unsqueeze(0).repeat_interleave(c.shape[0], dim=0)
        # c[:, :16] = cam2world_matrix.view(-1, 16)
        intrinsics = c[:, 16:25].reshape(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        H = W = self.neural_rendering_resolution
        # Create a batch of rays for volume rendering
        ray_origins, ray_directions, _ = self.ray_sampler(
            cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape

        # # Reshape output into three 32-channel planes
        # if planes.shape[1] == 3 * 2 * self.decoder_in_chans:
        #     # if isinstance(planes, tuple):
        #     #     N *= 2
        #     triplane_bg = True
        #     # planes = torch.cat(planes, 0) # inference in parallel
        #     # ray_origins = ray_origins.repeat(2,1,1)
        #     # ray_directions = ray_directions.repeat(2,1,1)

        # else:
        #     triplane_bg = False

        # assert not triplane_bg

        planes = planes.view(
            len(planes),
            3,
            -1,  # ! support background plane
            planes.shape[-2],
            planes.shape[-1])  # BS 96 256 256

        # Perform volume rendering
        rendering_details = self.renderer(planes,
                                          self.decoder,
                                          ray_origins,
                                          ray_directions,
                                          self.rendering_kwargs,
                                          return_meta=return_meta)

        feature_samples, depth_samples, weights_samples = (
            rendering_details[k]
            for k in ['feature_samples', 'depth_samples', 'weights_samples'])

        if return_sampling_details_flag:
            shape_synthesized = rendering_details['shape_synthesized']
        else:
            shape_synthesized = None

        # Reshape into 'raw' neural-rendered image
        feature_image = feature_samples.permute(0, 2, 1).reshape(
            N, feature_samples.shape[-1], H,
            W).contiguous()  # B 32 H W, in [-1,1]
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        weights_samples = weights_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        bcg_image = self.bcg_decoder(bg_plane)
        bcg_image = torch.nn.functional.interpolate(
            bcg_image,
            size=feature_image.shape[2:],
            mode='bilinear',
            align_corners=False,
            antialias=self.rendering_kwargs['sr_antialias'])

        mask_image = weights_samples * (1 + 2 * 0.001) - 0.001

        # ! fuse fg/bg model output
        feature_image = feature_image + (1 - weights_samples) * bcg_image

        rgb_image = feature_image[:, :3]

        # # Run superresolution to get final image
        if self.superresolution is not None and not return_raw_only:
            # assert ws is not None, 'feed in [cls] token here for SR module'

            if ws is not None and ws.ndim == 2:
                ws = ws.unsqueeze(
                    1)[:, -1:, :]  # follow stylegan tradition, B, N, C

            sr_image = self.superresolution(
                rgb=rgb_image,
                x=feature_image,
                base_x=rgb_image,
                ws=ws,  # only use the last layer
                noise_mode=self.
                rendering_kwargs['superresolution_noise_mode'],  # none
                **{
                    k: synthesis_kwargs[k]
                    for k in synthesis_kwargs.keys() if k != 'noise_mode'
                })
        else:
            # sr_image = rgb_image
            sr_image = None

        if shape_synthesized is not None:
            shape_synthesized.update({
                'image_depth': depth_image,
            })  # for 3D loss easy computation, wrap all 3D in a single dict

        ret_dict = {
            'feature_image': feature_image,
            # 'image_raw': feature_image[:, :3],
            'image_raw': rgb_image,
            'image_depth': depth_image,
            'weights_samples': weights_samples,
            # 'silhouette': mask_image,
            # 'silhouette_normalized_3channel': (mask_image*2-1).repeat_interleave(3,1), # N 3 H W
            'shape_synthesized': shape_synthesized,
            "image_mask": mask_image,
        }

        if sr_image is not None:
            ret_dict.update({
                'image_sr': sr_image,
            })

        if return_meta:
            ret_dict.update({
                'feature_volume':
                rendering_details['feature_volume'],
                'all_coords':
                rendering_details['all_coords'],
                'weights':
                rendering_details['weights'],
            })

        return ret_dict
