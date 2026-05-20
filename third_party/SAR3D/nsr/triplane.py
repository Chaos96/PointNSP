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
from utils.torch_utils import persistence
from .networks_stylegan2 import Generator as StyleGAN2Backbone
from .networks_stylegan2 import ToRGBLayer, SynthesisNetwork, MappingNetwork
from .volumetric_rendering.renderer import ImportanceRenderer
from .volumetric_rendering.ray_sampler import RaySampler, PatchRaySampler
import dnnlib
from pdb import set_trace as st
import math

import torch.nn.functional as F
import itertools
from ldm.modules.diffusionmodules.model import SimpleDecoder, Decoder

from Instantmesh.camera.perspective_camera import PerspectiveCamera
from Instantmesh.render.neural_render import NeuralRender
from Instantmesh.rep_3d.flexicubes_geometry import FlexiCubesGeometry
import numpy as np
import trimesh

def save_obj(pointnp_px3, facenp_fx3, colornp_px3, fpath):
    """Save mesh as OBJ file with vertex colors
    Args:
        pointnp_px3: Vertex positions (P x 3)
        facenp_fx3: Face indices (F x 3) 
        colornp_px3: Vertex colors (P x 3)
        fpath: Output file path
    """
    # Transform to match OBJ coordinate system
    pointnp_px3 = pointnp_px3 @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    facenp_fx3 = facenp_fx3[:, [2, 1, 0]]

    mesh = trimesh.Trimesh(
        vertices=pointnp_px3,
        faces=facenp_fx3,
        vertex_colors=colornp_px3,
    )
    mesh.export(fpath, 'obj')

def sdf_reg_loss_batch(sdf, all_edges):
    """Compute SDF regularization loss for batch of SDFs
    Args:
        sdf: SDF values (batch_size x num_points)
        all_edges: Edge indices connecting SDF points
    Returns:
        Loss encouraging consistent signs across edges
    """
    sdf_f1x6x2 = sdf[:, all_edges.reshape(-1)].reshape(sdf.shape[0], -1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    
    sdf_diff = F.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()) + \
               F.binary_cross_entropy_with_logits(
                   sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff

def camera_to_spherical_batch(c2ws):
    """Convert camera-to-world matrices to spherical coordinates
    Args:
        c2ws: Camera-to-world matrices (M x N x 4 x 4)
    Returns:
        Spherical coordinates (M x N x 3) as (radius, azimuth, elevation)
    """
    # Extract camera positions
    positions = c2ws[..., :3, 3]  # Shape: M x N x 3

    # Get x,y,z components
    x = positions[..., 0]
    y = positions[..., 1] 
    z = positions[..., 2]

    # Convert to spherical coordinates
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.atan2(y, x)  # Azimuth
    phi = torch.atan2(z, torch.sqrt(x**2 + y**2))  # Elevation

    # Convert angles to degrees
    theta = torch.rad2deg(theta)
    phi = torch.rad2deg(phi)

    return torch.stack((r, theta, phi), dim=-1)

def get_camera_poses(radius=2.5, elevation=30.0, azimuth=180.0):
    """Get single camera pose at specified spherical coordinates
    Args:
        radius: Distance from origin
        elevation: Elevation angle in degrees
        azimuth: Azimuth angle in degrees
    Returns:
        4x4 camera extrinsics matrix
    """
    elevation = np.deg2rad(elevation)
    azimuth = np.deg2rad(azimuth)

    # Calculate camera position
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)
    
    camera_positions = [[x, y, z]]
    camera_positions = torch.tensor(camera_positions, dtype=torch.float32)

    return center_looking_at_camera_pose(camera_positions)

def get_camera_poses_batch(radius=2.5, elevation=30.0, azimuth=None):
    """Get batch of camera poses at specified spherical coordinates
    Args:
        radius: Distance from origin
        elevation: Elevation angle in degrees
        azimuth: Azimuth angles in degrees
    Returns:
        Batch of 4x4 camera extrinsics matrices
    """
    elevation = np.deg2rad(elevation)
    azimuth = np.deg2rad(azimuth)

    # Calculate camera positions
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)

    camera_positions = np.stack([x, y, z], axis=-1)
    camera_positions = torch.tensor(camera_positions, dtype=torch.float32)

    return center_looking_at_camera_pose_batch(camera_positions)

def center_looking_at_camera_pose_batch(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
    """Create batch of OpenGL camera extrinsics matrices
    Args:
        camera_position: Camera positions (M x 3)
        look_at: Look-at points (M x 3 or 3), defaults to origin
        up_world: Up vectors (M x 3 or 3), defaults to z-axis
    Returns:
        Batch of 4x4 camera extrinsics matrices (M x 4 x 4)
    """
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32, device=camera_position.device)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32, device=camera_position.device)

    # Expand singleton dimensions
    if camera_position.ndim == 2:
        if look_at.ndim == 1:
            look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        if up_world.ndim == 1:
            up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    # Calculate camera coordinate axes
    z_axis = F.normalize(camera_position - look_at, dim=-1)
    
    # Handle degenerate case
    if torch.abs(torch.sum(z_axis * up_world, dim=-1)).max() > 0.999:
        print('Warning: camera up and z-axis are almost parallel')
        up_world = torch.tensor([0, -1, 0], dtype=torch.float32, device=camera_position.device)
        if camera_position.ndim == 2:
            up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    x_axis = F.normalize(torch.cross(up_world, z_axis, dim=-1), dim=-1)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1)

    # Construct extrinsics matrices
    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    return pad_camera_extrinsics_4x4(extrinsics)

def center_looking_at_camera_pose(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
    """Create single OpenGL camera extrinsics matrix
    Args:
        camera_position: Camera position (3 or M x 3)
        look_at: Look-at point (3), defaults to origin
        up_world: Up vector (3), defaults to z-axis
    Returns:
        4x4 camera extrinsics matrix
    """
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    # Calculate camera coordinate axes
    z_axis = F.normalize(camera_position - look_at, dim=-1)

    # Handle degenerate case
    if torch.abs(torch.dot(z_axis[0], up_world[0])) > 0.999:
        print('Warning: camera up and z-axis are almost parallel')
        up_world = torch.tensor([0, -1, 0], dtype=torch.float32).unsqueeze(0)

    x_axis = F.normalize(torch.linalg.cross(up_world, z_axis, dim=-1), dim=-1)
    y_axis = F.normalize(torch.linalg.cross(z_axis, x_axis, dim=-1), dim=-1)

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    return pad_camera_extrinsics_4x4(extrinsics)

def pad_camera_extrinsics_4x4(extrinsics):
    """Pad 3x4 camera extrinsics to 4x4 homogeneous matrix"""
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    return torch.cat([extrinsics, padding], dim=-2)

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
        rendering_details = self.renderer(
            planes,
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


@persistence.persistent_class
class OSGDecoder(torch.nn.Module):
    """Decoder that separately predicts RGB colors and density (sigma) values.
    
    This decoder takes sampled triplane features and outputs:
    1. RGB colors through a MLP with sigmoid activation
    2. Density values through a separate MLP
    """

    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        # Output dimensions for RGB colors and density
        self.decoder_output_dim_rgb = 3  # RGB channels
        self.decoder_output_dim_sigma = 1  # Density

        # MLP for predicting RGB colors
        self.net_rgb = torch.nn.Sequential(
            FullyConnectedLayer(n_features,
                                self.hidden_dim,
                                lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim,
                                3,
                                lr_multiplier=options['decoder_lr_mul']))
        # Activation for RGB output - sigmoid or leaky ReLU
        self.activation = options.get('decoder_activation', 'sigmoid')

        # MLP for predicting density values
        self.net_sigma = torch.nn.Sequential(
            FullyConnectedLayer(n_features,
                                self.hidden_dim,
                                lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim,
                                1,
                                lr_multiplier=options['decoder_lr_mul']))

    def forward(self, sampled_features, ray_directions):
        """Forward pass to predict RGB colors and densities.
        
        Args:
            sampled_features: Features sampled from triplanes [N, 3, M, C]
            ray_directions: Ray direction vectors [N, M, 3]
            
        Returns:
            dict: Contains 'rgb' [N, M, 3] and 'sigma' [N, M, 1] predictions
        """
        # Average features across triplane dimension
        sampled_features = sampled_features.mean(1)  # [N, M, C]
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N * M, C)  # Flatten batch and ray dimensions

        # Predict RGB colors
        x_rgb = self.net_rgb(x)
        x_rgb = x_rgb.view(N, M, -1)
        rgb = x_rgb[..., 0:]

        # Predict density values  
        x_sigma = self.net_sigma(x)
        x_sigma = x_sigma.view(N, M, -1)
        sigma = x_sigma[..., 0:]
        
        # Apply final activation to RGB values
        if self.activation == "sigmoid":
            # Original EG3D activation with small offset
            rgb = torch.sigmoid(rgb) * (1 + 2 * 0.001) - 0.001
        elif self.activation == "lrelu":
            # StyleGAN2-style activation
            rgb = torch.nn.functional.leaky_relu(rgb, 0.2,
                                                 inplace=True) * math.sqrt(2)
        return {'rgb': rgb, 'sigma': sigma}


@persistence.persistent_class
class OSGDecoderInstantMesh(nn.Module):
    """Triplane decoder for instant mesh generation.
    
    This decoder concatenates features from all triplanes and uses a single MLP
    to jointly predict RGB colors and density values.
    
    Reference:
    EG3D: https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py#L112
    """
    def __init__(self, n_features: int,
                 hidden_dim: int = 64, num_layers: int = 4, activation: nn.Module = nn.ReLU):
        super().__init__()
        # Single MLP for joint prediction
        self.net = nn.Sequential(
            nn.Linear(3 * n_features, hidden_dim),  # Concat features from 3 planes
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 1 + 3),  # Output density + RGB
        )
        # Initialize all biases to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, sampled_features, ray_directions):
        """Forward pass to predict RGB colors and densities.
        
        Args:
            sampled_features: Features sampled from triplanes [N, 3, M, C]
            ray_directions: Ray direction vectors [N, M, 3]
            
        Returns:
            dict: Contains 'rgb' [N, M, 3] and 'sigma' [N, M, 1] predictions
        """
        # Concatenate features from all triplanes
        _N, n_planes, _M, _C = sampled_features.shape
        sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(_N, _M, n_planes*_C)
        x = sampled_features

        N, M, C = x.shape
        x = x.contiguous().view(N*M, C)  # Flatten batch and ray dimensions

        # Joint prediction of density and RGB
        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001
        sigma = x[..., 0:1]

        return {'rgb': rgb, 'sigma': sigma}

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


def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
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
    plane_features = plane_features.view(N * n_planes, C, H, W)

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
    """Main triplane model for neural rendering.
    
    This module takes conditioning inputs and generates neural rendered images through:
    1. A triplane feature representation
    2. Volume rendering with importance sampling
    3. Optional super-resolution
    """

    def __init__(
        self,
        c_dim=25,                    # Conditioning label dimension
        img_resolution=128,          # Output image resolution
        img_channels=3,              # Number of output channels
        out_chans=96,               # Output feature channels
        triplane_size=224,          # Size of triplane feature maps
        rendering_kwargs={},         # Volume rendering settings
        decoder_in_chans=32,        # Input channels for decoder
        decoder_output_dim=32,      # Output dimension of decoder
        sr_num_fp16_res=0,          # Number of FP16 layers in super-resolution
        sr_kwargs={},               # Super-resolution settings
        create_triplane=False,      # Whether to create single triplane for overfitting
        bcg_synthesis_kwargs={},    # Background synthesis settings
        lrm_decoder=False,          # Whether to use LRM decoder
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.triplane_size = triplane_size
        self.decoder_in_chans = decoder_in_chans
        self.out_chans = out_chans

        # Initialize renderer and ray sampler
        self.renderer = ImportanceRenderer()
        self.ray_sampler = PatchRaySampler() if 'PatchRaySampler' in rendering_kwargs else RaySampler()

        if True:
            self.decoder = LRMOSGDecoder(decoder_in_chans)
        else:
            self.decoder = OSGDecoder(
                decoder_in_chans,
                {
                    'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                    'decoder_output_dim': decoder_output_dim
                })

        self.neural_rendering_resolution = img_resolution
        self.rendering_kwargs = rendering_kwargs
        
        # Create single triplane if needed
        self.create_triplane = create_triplane
        if create_triplane:
            self.planes = nn.Parameter(torch.randn(1, out_chans, 256, 256))

        # Initialize super-resolution if needed
        if bool(sr_kwargs):
            assert decoder_in_chans == decoder_output_dim, 'tradition'
            if rendering_kwargs['superresolution_module'] in [
                    'utils.torch_utils.components.PixelUnshuffleUpsample',
                    'utils.torch_utils.components.NearestConvSR',
                    'utils.torch_utils.components.NearestConvSR_Residual'
            ]:
                self.superresolution = dnnlib.util.construct_class_by_name(
                    class_name=rendering_kwargs['superresolution_module'],
                    sr_ratio=2,  # 2x upsampling
                    output_dim=decoder_output_dim,
                    num_out_ch=3,
                )
            else:
                self.superresolution = dnnlib.util.construct_class_by_name(
                    class_name=rendering_kwargs['superresolution_module'],
                    channels=decoder_output_dim,
                    img_resolution=img_resolution,
                    sr_num_fp16_res=sr_num_fp16_res,
                    sr_antialias=rendering_kwargs['sr_antialias'],
                    **sr_kwargs)
        else:
            self.superresolution = None

        self.bcg_synthesis = None

    def forward(
            self,
            planes=None,             # Input feature planes
            c=None,                  # Conditioning
            ws=None,                 # Style vectors
            ray_origins=None,        # Ray origins
            ray_directions=None,     # Ray directions  
            z_bcg=None,             # Background latent
            neural_rendering_resolution=None,  # Override resolution
            update_emas=False,       # Update EMA params
            cache_backbone=False,    # Cache backbone features
            use_cached_backbone=False,  # Use cached features
            return_meta=False,       # Return metadata
            return_raw_only=False,   # Skip super-resolution
            sample_ray_only=False,   # Only sample rays
            fg_bbox=None,            # Foreground bounding box
            **synthesis_kwargs):
        """Forward pass for neural rendering.

        Args:
            planes: Input feature planes
            c: Conditioning (camera params)
            ws: Style vectors for super-resolution
            ray_origins: Ray origin points
            ray_directions: Ray direction vectors
            z_bcg: Background latent code
            neural_rendering_resolution: Override rendering resolution
            update_emas: Update EMA parameters
            cache_backbone: Cache backbone features
            use_cached_backbone: Use cached features
            return_meta: Return additional metadata
            return_raw_only: Skip super-resolution
            sample_ray_only: Only sample rays
            fg_bbox: Foreground bounding box
            synthesis_kwargs: Additional synthesis args

        Returns:
            Dictionary containing:
            - Raw neural rendered image
            - Super-resolved image (optional)
            - Depth map
            - Feature maps
            - Weights/mask
            - Additional metadata (optional)
        """

        # Extract camera parameters
        cam2world_matrix = c[:, :16].reshape(-1, 4, 4)
        intrinsics = c[:, 16:25].reshape(-1, 3, 3)

        # Set rendering resolution
        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Generate rays if not provided
        if ray_directions is None:
            H = W = self.neural_rendering_resolution
            
            if sample_ray_only:
                # Sample rays for patch rendering
                ray_origins, ray_directions, ray_bboxes = self.ray_sampler(
                    cam2world_matrix, intrinsics,
                    self.rendering_kwargs.get('patch_rendering_resolution'),
                    self.neural_rendering_resolution, fg_bbox)

                return {
                    'ray_origins': ray_origins,
                    'ray_directions': ray_directions, 
                    'ray_bboxes': ray_bboxes,
                }

            else:
                # Sample rays for full rendering
                ray_origins, ray_directions, _ = self.ray_sampler(
                    cam2world_matrix, intrinsics, self.neural_rendering_resolution,
                    self.neural_rendering_resolution)

        else:
            H = W = int(ray_directions.shape[1]**0.5)

        # Use stored planes if none provided
        if planes is None:
            assert self.planes is not None
            planes = self.planes.repeat_interleave(c.shape[0], dim=0)

        return_sampling_details_flag = self.rendering_kwargs.get(
            'return_sampling_details_flag', False)

        if return_sampling_details_flag:
            return_meta = True

        # Get batch size
        N, M, _ = ray_origins.shape

        # Check if using background triplane
        if planes.shape[1] == 3 * 2 * self.decoder_in_chans:
            triplane_bg = True
        else:
            triplane_bg = False

        # Reshape planes
        planes = planes.reshape(
            len(planes),
            3,
            -1,
            planes.shape[-2],
            planes.shape[-1])

        # Perform volume rendering
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

        # Extract rendering outputs
        feature_samples, depth_samples, weights_samples = (
            rendering_details[k]
            for k in ['feature_samples', 'depth_samples', 'weights_samples'])

        # Get synthesized shape if needed
        shape_synthesized = rendering_details['shape_synthesized'] if return_sampling_details_flag else None

        # Reshape outputs to images
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        weights_samples = weights_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Create mask
        mask_image = weights_samples * (1 + 2 * 0.001) - 0.001

        # Composite with background if using background triplane
        if triplane_bg:
            feature_image = (1 - mask_image) * rendering_details['bg_ret_dict']['rgb_final'] + feature_image

        rgb_image = feature_image[:, :3]

        # Apply super-resolution if available
        if self.superresolution is not None and not return_raw_only:
            if ws is not None and ws.ndim == 2:
                ws = ws.unsqueeze(1)[:, -1:, :]

            sr_image = self.superresolution(
                rgb=rgb_image,
                x=feature_image,
                base_x=rgb_image,
                ws=ws,
                noise_mode=self.rendering_kwargs['superresolution_noise_mode'],
                **{k: synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'}
            )
        else:
            sr_image = None

        # Update shape info if needed
        if shape_synthesized is not None:
            shape_synthesized.update({'image_depth': depth_image})

        # Prepare return dictionary
        ret_dict = {
            'feature_image': feature_image,
            'image_raw': rgb_image,
            'image_depth': depth_image,
            'weights_samples': weights_samples,
            'shape_synthesized': shape_synthesized,
            "image_mask": mask_image,
        }

        if sr_image is not None:
            ret_dict.update({'image_sr': sr_image})

        if return_meta:
            ret_dict.update({
                'feature_volume': rendering_details['feature_volume'],
                'all_coords': rendering_details['all_coords'],
                'weights': rendering_details['weights'],
            })

        return ret_dict


# Class for generating and rendering 3D meshes from triplanes
class TriplaneMesh(torch.nn.Module):

    def __init__(
        self,
        c_dim=25,                # Conditioning label dimensionality
        img_resolution=128,      # Output image resolution
        img_channels=3,          # Number of output color channels
        out_chans=96,           # Output feature channels
        triplane_size=224,      # Size of triplane features
        rendering_kwargs={},     # Rendering options
        decoder_in_chans=32,    # Decoder input channels
        decoder_output_dim=32,  # Decoder output dimensions
        sr_num_fp16_res=0,      # Number of FP16 super-resolution layers
        sr_kwargs={},           # Super-resolution options
        create_triplane=False,  # Whether to create triplane for single instance overfitting
        bcg_synthesis_kwargs={}, # Background synthesis options
        lrm_decoder=False,      # Whether to use LRM decoder
        grid_res=64,           # Resolution of 3D grid
        grid_scale=2.1,        # Scale of 3D grid
    ):
        super().__init__()
        
        grid_res = 128  # Override grid resolution
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.triplane_size = triplane_size
        self.decoder_in_chans = decoder_in_chans
        self.out_chans = out_chans

        # Initialize renderers
        self.renderer = ImportanceRenderer()
        self.ray_sampler = PatchRaySampler() if 'PatchRaySampler' in rendering_kwargs else RaySampler()

        # Initialize decoder
        if lrm_decoder:
            self.decoder = LRMOSGDecoderMesh(decoder_in_chans)
        else:
            self.decoder = OSGDecoder(
                decoder_in_chans,
                {
                    'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                    'decoder_output_dim': decoder_output_dim
                })

        self.neural_rendering_resolution = img_resolution
        self.rendering_kwargs = rendering_kwargs
        
        # Optional triplane creation
        self.create_triplane = create_triplane
        if create_triplane:
            self.planes = nn.Parameter(torch.randn(1, out_chans, 256, 256))

        # Initialize super-resolution if needed
        if bool(sr_kwargs):
            assert decoder_in_chans == decoder_output_dim, 'tradition'
            if rendering_kwargs['superresolution_module'] in [
                    'utils.torch_utils.components.PixelUnshuffleUpsample',
                    'utils.torch_utils.components.NearestConvSR',
                    'utils.torch_utils.components.NearestConvSR_Residual'
            ]:
                self.superresolution = dnnlib.util.construct_class_by_name(
                    class_name=rendering_kwargs['superresolution_module'],
                    sr_ratio=2,  # 2x super-resolution
                    output_dim=decoder_output_dim,
                    num_out_ch=3,
                )
            else:
                self.superresolution = dnnlib.util.construct_class_by_name(
                    class_name=rendering_kwargs['superresolution_module'],
                    channels=decoder_output_dim,
                    img_resolution=img_resolution,
                    sr_num_fp16_res=sr_num_fp16_res,
                    sr_antialias=rendering_kwargs['sr_antialias'],
                    **sr_kwargs)
        else:
            self.superresolution = None

        self.bcg_synthesis = None

        # Mesh generation parameters
        self.grid_res = grid_res
        self.grid_scale = grid_scale
        self.deformation_multiplier = 4.0

    def init_flexicubes_geometry(self, device, fovy=50.0):
        """Initialize FlexiCubes geometry with camera and renderer"""
        camera = PerspectiveCamera(fovy=fovy, device=device)
        renderer = NeuralRender(device, camera_model=camera)
        self.geometry = FlexiCubesGeometry(
            grid_res=self.grid_res, 
            scale=self.grid_scale, 
            renderer=renderer, 
            render_type='neural_render',
            device=device,
        )
    
    def forward(
            self,
            planes=None,
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
        """Forward pass to generate mesh and render images"""

        # Process camera matrices
        if c.shape[0] == 1 and c.shape[1] == 25:
            c = c.unsqueeze(0)
            render_c2ws = c[:,:,:16].reshape(-1, 1, 4, 4)
        else:
            render_c2ws = c[:,:,:16].reshape(-1, 6, 4, 4)
            
        # Transform camera matrices
        transform_R = torch.tensor([[1,0,0], [0,-1,0],[0,0,-1]], dtype=render_c2ws.dtype).to(render_c2ws.device)
        render_c2ws[:,:,:3,:3] = torch.matmul(render_c2ws[:,:,:3,:3], transform_R.unsqueeze(0).unsqueeze(0))
        render_c2ws[:,:,:3,3] = 2.0 * render_c2ws[:,:,:3,3]
        render_w2cs = torch.linalg.inv(render_c2ws.float())

        # Reshape planes
        planes = planes.reshape(
            len(planes),
            3,
            -1,  # ! support background plane
            planes.shape[-2],
            planes.shape[-1])

        # Generate geometry and render
        out = self.forward_geometry(
            planes = planes,
            render_cameras = render_w2cs,
            render_size = self.neural_rendering_resolution
        )

        return out
       
    
    def forward_geometry(self, planes, render_cameras, render_size=256):
        """Main function to generate 3D mesh and render it to 2D images.

        Args:
            planes: Triplane features
            render_cameras: World-to-camera transformation matrices 
            render_size: Output render resolution
        """
        B, NV = render_cameras.shape[:2]  # Batch size, number of views

        # Handle single plane case
        if planes.shape[0] == 12:
            planes = planes[0].unsqueeze(0)

        # Generate 3D mesh
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(planes)

        # Handle multi-plane case
        if planes.shape[0] != 1:
            mesh_v = mesh_v + mesh_v
            mesh_f = mesh_f + mesh_f

        # Render mesh to 2D
        cam_mv = render_cameras
        run_n_view = cam_mv.shape[1]
        
        # Get render outputs
        antilias_mask, hard_mask, tex_pos, depth, normal = self.render_mesh(
            mesh_v, mesh_f, cam_mv, render_size=render_size)

        tex_hard_mask = hard_mask

        # Reshape texture positions and masks
        tex_pos = [torch.cat([pos[i_view:i_view + 1] for i_view in range(run_n_view)], dim=2) for pos in tex_pos]
        tex_hard_mask = torch.cat(
            [torch.cat(
                [tex_hard_mask[i * run_n_view + i_view: i * run_n_view + i_view + 1]
                 for i_view in range(run_n_view)], dim=2)
                for i in range(planes.shape[0] * 2)], dim=0)

        # Get texture features
        if planes.shape[0] == 1:
            tex_feat = self.get_texture_prediction(planes, tex_pos, tex_hard_mask)
        else:
            tex_feat = self.get_texture_prediction(planes.repeat(2,1,1,1,1), tex_pos, tex_hard_mask)

        # White background
        background_feature = torch.ones_like(tex_feat)

        # Composite foreground and background
        img_feat = tex_feat * tex_hard_mask + background_feature * (1 - tex_hard_mask)

        # Reshape back to image format
        img_feat = torch.cat(
            [torch.cat(
                [img_feat[i:i + 1, :, render_size * i_view: render_size * (i_view + 1)]
                 for i_view in range(run_n_view)], dim=0) for i in range(len(tex_pos))], dim=0)

        # Format outputs
        img = img_feat.clamp(0, 1).permute(0, 3, 1, 2)
        img = img * 2 - 1  # Scale to [-1,1]
        antilias_mask = antilias_mask.permute(0, 3, 1, 2)
        depth = -depth.permute(0, 3, 1, 2)
        normal = normal * 2 - 1
        normal = normal.permute(0, 3, 1, 2)

        # Calculate regularization losses
        sdf_reg_loss_entropy = sdf_reg_loss_batch(sdf, self.geometry.all_edges).mean() * 0.01
        _, flexicubes_surface_reg, flexicubes_weight_reg = sdf_reg_loss
        flexicubes_surface_reg = flexicubes_surface_reg.mean() * 0.5
        flexicubes_weight_reg = flexicubes_weight_reg.mean() * 0.1

        return {
            'image_raw': img,
            'image_mask': antilias_mask,
            'image_depth_mesh': depth,
            'image_normal_mesh': normal,
            'sdf_reg_loss_entropy': sdf_reg_loss_entropy,
            'flexicubes_surface_reg': flexicubes_surface_reg,
            'flexicubes_weight_reg': flexicubes_weight_reg,
        }
    
    def render_mesh(self, mesh_v, mesh_f, cam_mv, render_size=256):
        """Render mesh using nvdiffrast.

        Args:
            mesh_v: List of mesh vertices
            mesh_f: List of mesh faces
            cam_mv: Camera view matrices
            render_size: Output render resolution
        """
        return_value_list = []
        
        for i_mesh in range(len(mesh_v)):
            return_value = self.geometry.render_mesh(
                mesh_v[i_mesh],
                mesh_f[i_mesh].int(),
                cam_mv[i_mesh],
                resolution=render_size,
                hierarchical_mask=False
            )
            return_value_list.append(return_value)

        # Combine outputs from multiple meshes
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
        """Predict texture features at given positions.

        Args:
            planes: Triplane features
            tex_pos: Query positions for texture
            hard_mask: Silhouette mask
        """
        tex_pos = torch.cat(tex_pos, dim=0)
        if hard_mask is not None:
            tex_pos = tex_pos * hard_mask.float()
        batch_size = tex_pos.shape[0]
        tex_pos = tex_pos.reshape(batch_size, -1, 3)

        # Use mask to get texture locations (memory optimization)
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

        # Get texture features
        tex_feat = torch.utils.checkpoint.checkpoint(
            self.decoder.get_texture_prediction,
            planes, 
            tex_pos,
            use_reentrant=False,
        )

        # Reshape texture features
        if hard_mask is not None:
            final_tex_feat = torch.zeros(
                planes.shape[0], hard_mask.shape[1] * hard_mask.shape[2], tex_feat.shape[-1], 
                device=tex_feat.device).to(tex_feat.dtype)
            expanded_hard_mask = hard_mask.reshape(hard_mask.shape[0], -1, 1).expand(-1, -1, final_tex_feat.shape[-1]) > 0.5
            
            for i in range(planes.shape[0]):
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
        """Extract 3D mesh from FlexiCubes. Only supports batch size 1.

        Args:
            planes: Triplane features
            use_texture_map: Whether to use texture map or vertex colors
            texture_resolution: Resolution of texture map
        """
        assert planes.shape[0] == 1
        device = planes.device

        # Get geometry
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(planes)
        vertices, faces = mesh_v[0], mesh_f[0]

        # Use vertex colors
        if not use_texture_map:
            vertices_tensor = vertices.unsqueeze(0)
            vertices_colors = self.decoder.get_texture_prediction(
                planes, vertices_tensor.float()).float().clamp(0, 1).squeeze(0).cpu().numpy()
            vertices_colors = (vertices_colors * 255).astype(np.uint8)
            return vertices.float().cpu().numpy(), faces.cpu().numpy(), vertices_colors

        # Use texture map with UV mapping
        import nvdiffrast.torch as dr
        from Instantmesh.rep_3d.extract_texture_map import xatlas_uvmap
        ctx = dr.RasterizeCudaContext(device=device)
        uvs, mesh_tex_idx, gb_pos, tex_hard_mask = xatlas_uvmap(
            self.geometry.renderer.ctx, vertices, faces, resolution=texture_resolution)
        tex_hard_mask = tex_hard_mask.float()

        # Get texture map colors
        tex_feat = self.get_texture_prediction(
            planes, [gb_pos], tex_hard_mask)
        background_feature = torch.zeros_like(tex_feat)
        img_feat = torch.lerp(background_feature, tex_feat, tex_hard_mask)
        texture_map = img_feat.permute(0, 3, 1, 2).squeeze(0)

        return vertices, faces, uvs, mesh_tex_idx, texture_map
    
    def get_geometry_prediction(self, planes=None):
        """Generate mesh from triplane features.

        Args:
            planes: Triplane feature maps
        """
        # Get SDF and deformation values
        sdf, deformation, sdf_reg_loss, weight = self.get_sdf_deformation_prediction(planes)
        v_deformed = self.geometry.verts.to(deformation.dtype).unsqueeze(dim=0).expand(sdf.shape[0], -1, -1) + deformation
        tets = self.geometry.indices
        n_batch = planes.shape[0]
        v_list = []
        f_list = []
        flexicubes_surface_reg_list = []

        # Generate mesh using marching tetrahedra
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
        """Predict SDF and deformation for tetrahedron vertices.

        Args:
            planes: Triplane feature maps
        """
        init_position = self.geometry.verts.unsqueeze(0).expand(planes.shape[0], -1, -1)
        
        # Get SDF and deformation predictions
        sdf, deformation, weight = torch.utils.checkpoint.checkpoint(
            self.decoder.get_geometry_prediction,
            planes, 
            init_position, 
            self.geometry.indices,
            use_reentrant=False,
        )

        # Normalize deformation
        deformation = 1.0 / (self.grid_res * self.deformation_multiplier) * torch.tanh(deformation)
        sdf_reg_loss = torch.zeros(sdf.shape[0], device=sdf.device, dtype=torch.float32)

        # Handle empty shapes
        sdf_bxnxnxn = sdf.reshape((sdf.shape[0], self.grid_res + 1, self.grid_res + 1, self.grid_res + 1))
        sdf_less_boundary = sdf_bxnxnxn[:, 1:-1, 1:-1, 1:-1].reshape(sdf.shape[0], -1)
        pos_shape = torch.sum((sdf_less_boundary > 0).int(), dim=-1)
        neg_shape = torch.sum((sdf_less_boundary < 0).int(), dim=-1)
        zero_surface = torch.bitwise_or(pos_shape == 0, neg_shape == 0)

        if torch.sum(zero_surface).item() > 0:
            update_sdf = torch.zeros_like(sdf[0:1])
            max_sdf = sdf.max()
            min_sdf = sdf.min()
            update_sdf[:, self.geometry.center_indices] += (1.0 - min_sdf)  # Greater than zero
            update_sdf[:, self.geometry.boundary_indices] += (-1 - max_sdf)  # Smaller than zero
            new_sdf = torch.zeros_like(sdf)
            for i_batch in range(zero_surface.shape[0]):
                if zero_surface[i_batch]:
                    new_sdf[i_batch:i_batch + 1] += update_sdf
            update_mask = (new_sdf == 0).float()
            
            # Regularization to avoid fully positive/negative SDFs
            sdf_reg_loss = torch.abs(sdf).mean(dim=-1).mean(dim=-1)
            sdf_reg_loss = sdf_reg_loss * zero_surface.float()
            sdf = sdf * update_mask + new_sdf * (1 - update_mask)

        # Remove gradients for bad SDFs
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
    """Triplane model with separate foreground and background planes.
    
    This class extends the base Triplane model to handle separate background planes
    for improved background rendering quality.
    """

    def __init__(self,
                 c_dim=25,                    # Conditioning label dimension
                 img_resolution=128,          # Output image resolution
                 img_channels=3,              # Number of output channels
                 out_chans=96,               # Output feature channels
                 triplane_size=224,          # Size of triplane features
                 rendering_kwargs={},         # Volume rendering settings
                 decoder_in_chans=32,        # Input channels for decoder
                 decoder_output_dim=32,      # Output dimension of decoder
                 sr_num_fp16_res=0,          # Number of FP16 layers in super-resolution
                 sr_kwargs={},               # Super-resolution settings
                 bcg_synthesis_kwargs={}):    # Background synthesis settings
        
        super().__init__(c_dim, img_resolution, img_channels, out_chans,
                         triplane_size, rendering_kwargs, decoder_in_chans,
                         decoder_output_dim, sr_num_fp16_res, sr_kwargs,
                         bcg_synthesis_kwargs)

        # Initialize background decoder with reduced channel size
        self.bcg_decoder = Decoder(
            ch=64,                  # Half channel size for efficiency
            out_ch=32,
            ch_mult=(1, 2),         # Use resolution of 64 for background
            num_res_blocks=2,
            dropout=0.0,
            attn_resolutions=(),
            z_channels=4,
            resolution=64,
            in_channels=3,
        )

    def forward(
            self,
            planes,                         # Foreground feature planes
            bg_plane,                       # Background feature plane
            c,                              # Camera parameters
            ws=None,                        # Style vectors
            z_bcg=None,                     # Background latent code
            neural_rendering_resolution=None,# Optional override for render resolution
            update_emas=False,              # Whether to update EMA weights
            cache_backbone=False,           # Whether to cache backbone features
            use_cached_backbone=False,      # Whether to use cached features
            return_meta=False,              # Whether to return additional metadata
            return_raw_only=False,          # Whether to skip super-resolution
            **synthesis_kwargs):            # Additional synthesis options
        """Forward pass to generate neural rendered images with separate background.

        Args:
            planes: Foreground triplane features
            bg_plane: Background plane features
            c: Camera parameters including pose and intrinsics
            ws: Optional style vectors for super-resolution
            Additional args control rendering behavior and output format

        Returns:
            Dictionary containing rendered images and optional metadata
        """

        # Use stored planes if none provided
        if planes is None:
            assert self.planes is not None
            planes = self.planes.repeat_interleave(c.shape[0], dim=0)

        # Check if detailed sampling info requested
        return_sampling_details_flag = self.rendering_kwargs.get(
            'return_sampling_details_flag', False)
        if return_sampling_details_flag:
            return_meta = True

        # Extract camera matrices from input parameters
        cam2world_matrix = c[:, :16].reshape(-1, 4, 4)
        intrinsics = c[:, 16:25].reshape(-1, 3, 3)

        # Set rendering resolution
        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        H = W = self.neural_rendering_resolution

        # Generate rays for volume rendering
        ray_origins, ray_directions, _ = self.ray_sampler(
            cam2world_matrix, intrinsics, neural_rendering_resolution)

        N, M, _ = ray_origins.shape

        # Reshape planes into proper format
        planes = planes.view(
            len(planes),
            3,
            -1,
            planes.shape[-2],
            planes.shape[-1])

        # Perform volume rendering
        rendering_details = self.renderer(planes,
                                          self.decoder,
                                          ray_origins,
                                          ray_directions,
                                          self.rendering_kwargs,
                                          return_meta=return_meta)

        # Extract rendering outputs
        feature_samples, depth_samples, weights_samples = (
            rendering_details[k]
            for k in ['feature_samples', 'depth_samples', 'weights_samples'])

        # Get shape info if requested
        shape_synthesized = rendering_details['shape_synthesized'] if return_sampling_details_flag else None

        # Reshape rendered features into image format
        feature_image = feature_samples.permute(0, 2, 1).reshape(
            N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        weights_samples = weights_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Generate background image
        bcg_image = self.bcg_decoder(bg_plane)
        bcg_image = torch.nn.functional.interpolate(
            bcg_image,
            size=feature_image.shape[2:],
            mode='bilinear',
            align_corners=False,
            antialias=self.rendering_kwargs['sr_antialias'])

        # Generate foreground mask
        mask_image = weights_samples * (1 + 2 * 0.001) - 0.001

        # Composite foreground and background
        feature_image = feature_image + (1 - weights_samples) * bcg_image
        rgb_image = feature_image[:, :3]

        # Apply super-resolution if available
        if self.superresolution is not None and not return_raw_only:
            if ws is not None and ws.ndim == 2:
                ws = ws.unsqueeze(1)[:, -1:, :]

            sr_image = self.superresolution(
                rgb=rgb_image,
                x=feature_image,
                base_x=rgb_image,
                ws=ws,
                noise_mode=self.rendering_kwargs['superresolution_noise_mode'],
                **{k: synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'}
            )
        else:
            sr_image = None

        # Update shape info if available
        if shape_synthesized is not None:
            shape_synthesized.update({'image_depth': depth_image})

        # Prepare return dictionary
        ret_dict = {
            'feature_image': feature_image,
            'image_raw': rgb_image,
            'image_depth': depth_image,
            'weights_samples': weights_samples,
            'shape_synthesized': shape_synthesized,
            "image_mask": mask_image,
        }

        if sr_image is not None:
            ret_dict.update({'image_sr': sr_image})

        if return_meta:
            ret_dict.update({
                'feature_volume': rendering_details['feature_volume'],
                'all_coords': rendering_details['all_coords'],
                'weights': rendering_details['weights'],
            })

        return ret_dict
