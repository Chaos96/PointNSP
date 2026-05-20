"""
Utility functions for rendering 3D content from tri-plane representations.
Includes functions for video rendering, mesh extraction and visualization.
"""

# Standard library imports
import os

# Deep learning imports 
import torch

# 3D processing imports
import mcubes
import trimesh

# Visualization imports
import numpy as np
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

# Optimization imports
from xformers.triton import FusedLayerNorm as LayerNorm
from xformers.components.activations import Activation
from xformers.components.feedforward import fused_mlp

# Diffusion model imports
from guided_diffusion import dist_util

@torch.inference_mode()
def render_video_given_triplane(planes,
                              rec_model,
                              name_prefix='0', 
                              save_img=False,
                              render_reference=None,
                              save_mesh=False,
                              save_path="./sample_save"):
    """
    Render video from tri-plane representation with optional mesh extraction.
    
    Args:
        planes: Input tri-plane features
        rec_model: Reconstruction model
        name_prefix: Prefix for output files
        save_img: Whether to save individual frames
        render_reference: Reference data for rendering
        save_mesh: Whether to extract and save 3D mesh
        save_path: Output directory path
    """
    # Initialize pooling layers for different resolutions
    pool_128 = torch.nn.AdaptiveAvgPool2d((128, 128))
    pool_512 = torch.nn.AdaptiveAvgPool2d((512, 512))

    # Convert planes to float32 latents
    ddpm_latent = {'latent_after_vit': planes.to(torch.float32)}

    # Extract and save mesh if requested
    if save_mesh:
        mesh_size = 192
        mesh_thres = 10
        os.makedirs(save_path, exist_ok=True)
        dump_path = f'{save_path}/mesh/'
        os.makedirs(dump_path, exist_ok=True)
        
        # Generate 3D grid from tri-planes
        grid_out = rec_model(
            latent=ddpm_latent,
            grid_size=mesh_size,
            behaviour='triplane_decode_grid',
        )
        
        # Extract mesh using marching cubes
        vtx, faces = mcubes.marching_cubes(
            grid_out['sigma'].to(torch.float32).squeeze(0).squeeze(-1).cpu().numpy(),
            mesh_thres)
        
        # Scale vertices to world coordinates
        grid_scale = [rec_model.decoder.rendering_kwargs['sampler_bbox_min'], 
                     rec_model.decoder.rendering_kwargs['sampler_bbox_max']]
        vtx = (vtx / (mesh_size-1) * 2 - 1 ) * grid_scale[1]

        # Get vertex colors from tri-plane features
        vtx_tensor = torch.tensor(vtx, dtype=torch.float32, device=dist_util.dev()).unsqueeze(0)
        vtx_colors = rec_model.decoder.forward_points(ddpm_latent['latent_after_vit'], vtx_tensor)['rgb'].squeeze(0).cpu().numpy()
        vtx_colors = (vtx_colors.clip(0,1) * 255).astype(np.uint8)

        # Save colored mesh
        mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=vtx_colors)
        mesh_dump_path = os.path.join(dump_path, f'{name_prefix}.ply')
        mesh.export(mesh_dump_path, 'ply')
        print(f"Mesh dumped to {dump_path}")
        
        del grid_out, mesh
        torch.cuda.empty_cache()

    # Initialize video writer
    video_out = imageio.get_writer(
        f'{save_path}/triplane_{name_prefix}.mp4',
        mode='I',
        fps=15,
        codec='libx264')

    # Validate and process render reference
    if render_reference is None:
        raise ValueError('render_reference is None')
    else:
        for key in ['ins', 'bbox', 'caption']:
            if key in render_reference:
                render_reference.pop(key)

        render_reference = [{k: v[idx:idx + 1] for k, v in render_reference.items()} 
                          for idx in range(len(next(iter(render_reference.values()))))]

    # Render frames
    for i, batch in enumerate(tqdm(render_reference)):
        # Move batch to device
        micro = {k: v.to(dist_util.dev()) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}
        
        # Generate frame from tri-planes
        pred = rec_model(
            latent={'latent_after_vit': ddpm_latent['latent_after_vit'].repeat_interleave(6, dim=0).to(torch.float32).repeat(2,1,1,1)},
            c=micro['c'],
            behaviour='triplane_dec')
        
        # Process depth map for visualization
        pred_depth = pred['image_depth']
        pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
        pred_depth = pred_depth.cpu()[0].permute(1, 2, 0).numpy()
        pred_depth = (plt.cm.viridis(pred_depth[..., 0])[..., :3]) * 2 - 1
        pred_depth = torch.from_numpy(pred_depth).to(pred['image_raw'].device).permute(2, 0, 1).unsqueeze(0)

        # Handle different output resolutions
        if 'image_sr' in pred:
            gen_img = pred['image_sr']
            if pred['image_sr'].shape[-1] == 512:
                pred_vis = torch.cat([micro['img_sr'],
                                    pool_512(pred['image_raw']), gen_img,
                                    pool_512(pred_depth).repeat_interleave(3, dim=1)], dim=-1)
            elif pred['image_sr'].shape[-1] == 128:
                pred_vis = torch.cat([micro['img_sr'],
                                    pool_128(pred['image_raw']), pred['image_sr'],
                                    pool_128(pred_depth).repeat_interleave(3, dim=1)], dim=-1)
        else:
            gen_img = pred['image_raw']
            pred_vis = torch.cat([gen_img, pred_depth], dim=-1)

        # Save individual frames if requested
        if save_img:
            for batch_idx in range(gen_img.shape[0]):
                from PIL import Image
                sampled_img = Image.fromarray(
                    (gen_img[batch_idx].permute(1, 2, 0).cpu().numpy() *
                     127.5 + 127.5).clip(0, 255).astype(np.uint8))
                sampled_img.save(save_path + '/{}.png'.format(i))

        # Write frame to video
        vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
        vis = vis * 127.5 + 127.5
        vis = vis.clip(0, 255).astype(np.uint8)
        for j in range(vis.shape[0]):
            video_out.append_data(vis[j])

    video_out.close()
    print('Logged video to: ', f'{save_path}/triplane_{name_prefix}.mp4')

    # Clean up
    del video_out, vis, pred_vis, micro, pred

def render_video_given_triplane_mesh(planes,
                                   rec_model,
                                   name_prefix='0',
                                   save_img=False, 
                                   render_reference=None,
                                   save_mesh=False,
                                   save_path="./sample_save"):
    """
    Render video from tri-plane representation with mesh-based rendering.
    Similar to render_video_given_triplane but uses mesh-based rendering pipeline.
    
    Args: Same as render_video_given_triplane
    """
    pool_128 = torch.nn.AdaptiveAvgPool2d((128, 128))
    pool_512 = torch.nn.AdaptiveAvgPool2d((512, 512))

    ddpm_latent = {'latent_after_vit': planes.to(torch.float32)}
    os.makedirs(save_path, exist_ok=True)
    dump_path = f'{save_path}/mesh/'
    os.makedirs(dump_path, exist_ok=True)

    # Extract and save mesh if requested
    if save_mesh:
        planes = planes.reshape(len(planes), 3, -1, planes.shape[-2], planes.shape[-1])
        mesh_out = rec_model.decoder.triplane_decoder.extract_mesh(planes.float(), use_texture_map=False)
        vertices, faces, vertex_colors = mesh_out
        mesh_dump_path = os.path.join(dump_path, f'{name_prefix}.obj')
        save_obj(vertices, faces, vertex_colors, mesh_dump_path)
        print(f"Mesh dumped to {dump_path}")

    # Initialize video writer
    video_out = imageio.get_writer(
        f'{save_path}/triplane_{name_prefix}.mp4',
        mode='I',
        fps=15,
        codec='libx264')

    # Process render reference
    if render_reference is None:
        raise ValueError('render_reference is None')
    else:
        for key in ['ins', 'bbox', 'caption']:
            if key in render_reference:
                render_reference.pop(key)
        render_reference = [{k: v[idx:idx + 1] for k, v in render_reference.items()} 
                          for idx in range(len(next(iter(render_reference.values()))))]

    # Render frames
    for i, batch in enumerate(tqdm(render_reference)):
        micro = {k: v.to(dist_util.dev()) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}
        
        pred = rec_model(
            latent={'latent_after_vit': ddpm_latent['latent_after_vit'].repeat_interleave(6, dim=0).repeat(2,1,1,1)},
            c=micro['c'],
            behaviour='triplane_dec')

        # Process normal and depth maps
        pred_normal = pred['image_normal_mesh']
        pred_depth = pred['image_depth_mesh']
        pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
        pred_depth = pred_depth.cpu()[0].permute(1, 2, 0).numpy()
        pred_depth = (plt.cm.viridis(pred_depth[..., 0])[..., :3]) * 2 - 1
        pred_depth = torch.from_numpy(pred_depth).to(pred['image_raw'].device).permute(2, 0, 1).unsqueeze(0)

        # Handle different output resolutions
        if 'image_sr' in pred:
            gen_img = pred['image_sr']
            if pred['image_sr'].shape[-1] == 512:
                pred_vis = torch.cat([micro['img_sr'],
                                    pool_512(pred['image_raw']), gen_img,
                                    pool_512(pred_depth).repeat_interleave(3, dim=1)], dim=-1)
            elif pred['image_sr'].shape[-1] == 128:
                pred_vis = torch.cat([micro['img_sr'],
                                    pool_128(pred['image_raw']), pred['image_sr'],
                                    pool_128(pred_depth).repeat_interleave(3, dim=1)], dim=-1)
        else:
            gen_img = pred['image_raw']
            pred_vis = torch.cat([gen_img, pred_normal, pred_depth], dim=-1)

        # Save individual frames if requested
        if save_img:
            from PIL import Image
            for batch_idx in range(gen_img.shape[0]):
                sampled_img = Image.fromarray(
                    (gen_img[batch_idx].permute(1, 2, 0).cpu().numpy() *
                     127.5 + 127.5).clip(0, 255).astype(np.uint8))
                sampled_normal = Image.fromarray(
                    (pred_normal[batch_idx].permute(1, 2, 0).cpu().numpy() *
                     127.5 + 127.5).clip(0, 255).astype(np.uint8))
                sampled_normal.save(save_path + '/{}_normal.png'.format(i))
                sampled_img.save(save_path + '/{}.png'.format(i))

        # Write frame to video
        vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
        vis = vis * 127.5 + 127.5
        vis = vis.clip(0, 255).astype(np.uint8)
        for j in range(vis.shape[0]):
            video_out.append_data(vis[j])

    video_out.close()
    print('Logged video to: ', f'{save_path}/triplane_{name_prefix}.mp4')

    # Clean up
    del video_out, vis, pred_vis, micro, pred

def save_obj(pointnp_px3, facenp_fx3, colornp_px3, fpath):
    """
    Save a 3D mesh as an OBJ file.
    
    Args:
        pointnp_px3: Vertex coordinates (Nx3)
        facenp_fx3: Face indices (Mx3)
        colornp_px3: Vertex colors (Nx3)
        fpath: Output file path
    """
    # Transform coordinates and faces for correct orientation
    pointnp_px3 = pointnp_px3 @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    facenp_fx3 = facenp_fx3[:, [2, 1, 0]]

    mesh = trimesh.Trimesh(
        vertices=pointnp_px3,
        faces=facenp_fx3,
        vertex_colors=colornp_px3,
    )
    mesh.export(fpath, 'obj')