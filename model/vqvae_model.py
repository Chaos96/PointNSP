import torch
import torch.nn as nn
import torch.nn.functional as F

# Chamfer distance calculation
def chamfer_distance(x, y):
    xx = torch.sum(x**2, dim=2)
    yy = torch.sum(y**2, dim=2)
    zz = torch.matmul(x, y.transpose(2, 1))
    rx = xx.unsqueeze(2).expand(-1, -1, y.size(1))
    ry = yy.unsqueeze(1).expand(-1, x.size(1), -1)
    P = rx + ry - 2*zz
    return torch.mean(torch.min(P, dim=2)[0]) + torch.mean(torch.min(P, dim=1)[0])

class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, codebook_size, embedding_dim, num_points):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.vq_layer = VectorQuantizer(codebook_size, embedding_dim)  # Shared codebook
        self.decoder = Decoder(hidden_dim, input_dim, 5)  # 5 scales
        self.phi = nn.ModuleList([
            ResidualConv(hidden_dim, hidden_dim)
            for _ in range(5)  # 5 scales
        ])
        self.num_scales = 5
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_points = num_points

        self.upsample_factors = [4, 8, 8, 8]  # for 5 scales (1, 4, 32, 256, 2048)
        self.scale_points = [1, 4, 32, 256, 2048]

        # 3D convolution for downsampling
        self.downsample_conv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=2, stride=2)

        # MLPs for upsampling
        self.upsample_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * factor)
            ) for factor in self.upsample_factors
        ])

    def point_cloud_to_voxel(self, x, resolution):
        """Convert point cloud to voxel grid"""
        B, N, C = x.shape
        voxel = torch.zeros(B, C, resolution, resolution, resolution, device=x.device)
        
        # Normalize point cloud to [0, resolution-1]
        x_normalized = (x - x.min(dim=1, keepdim=True)[0]) / (x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0]) * (resolution - 1)
        x_indices = x_normalized.long()

        # Clip indices to ensure they are within the valid range
        x_indices = torch.clamp(x_indices, 0, resolution - 1)

        # Flatten the voxel grid
        flat_voxel = voxel.view(B, C, -1)

        # Create a flattened index
        flat_indices = x_indices[:, :, 0] * (resolution * resolution) + x_indices[:, :, 1] * resolution + x_indices[:, :, 2]

        # Use scatter_add_ to populate the voxel grid
        flat_voxel.scatter_add_(2, flat_indices.unsqueeze(1).expand(-1, C, -1), x.transpose(1, 2))

        return voxel

    def voxel_to_point_cloud(self, voxel, num_points):
        """Convert voxel grid back to point cloud"""
        B, C, D, H, W = voxel.shape
        points = voxel.reshape(B, C, -1).transpose(1, 2)
        
        # Sample points if necessary
        if points.shape[1] > num_points:
            idx = torch.randperm(points.shape[1])[:num_points]
            points = points[:, idx, :]
        
        return points

    def point_cloud_downsample(self, x, target_points):
        """
        Downsample the point cloud using 3DCNN
        """
        B, N, C = x.shape
        resolution = int(N ** (1/3))  # Assuming cubic voxel grid
        
        # Convert point cloud to voxel
        voxel = self.point_cloud_to_voxel(x, resolution)
        
        # Apply 3D convolution for downsampling
        downsampled_voxel = self.downsample_conv(voxel)
        
        # Convert back to point cloud
        downsampled_pc = self.voxel_to_point_cloud(downsampled_voxel, target_points)
        
        return downsampled_pc

    def point_cloud_upsample(self, x, target_size):
        """
        Upsample the point cloud using copying and MLP
        """
        B, N, C = x.shape
        if N >= target_size:
            return x[:, :target_size, :]

        # Apply MLP and upsample iteratively
        while N < target_size:
            # Find the closest upsampling factor
            factor = min(self.upsample_factors, key=lambda f: abs(f - (target_size // N)))
            mlp_index = self.upsample_factors.index(factor)
            mlp = self.upsample_mlps[mlp_index]

            # Apply MLP
            x_transformed = mlp(x)

            # Reshape and duplicate points
            x_reshaped = x_transformed.view(B, N, factor, -1)
            x = x_reshaped.view(B, N * factor, -1)
            
            N = x.shape[1]

        # Return the upsampled point cloud
        return x[:, :target_size, :]

    def encode(self, x):
        f = self.encoder(x)
        R = []
        vq_losses = []
        
        for k in range(self.num_scales):
            # import pdb; pdb.set_trace()
            current_points = self.scale_points[k]
            
            # Downsample f for each scale
            f_k = self.point_cloud_downsample(f, current_points)
            
            # Ensure f_k is contiguous and has the correct shape
            f_k = f_k.contiguous()
            if f_k.dim() == 2:
                f_k = f_k.unsqueeze(0)
            
            # Quantize
            quantized, vq_loss, encoding_indices = self.vq_layer(f_k)
            R.append(encoding_indices)
            vq_losses.append(vq_loss)
            
            # If not the last scale, prepare for the next iteration
            if k < self.num_scales - 1:
                z_k = self.point_cloud_upsample(quantized, f.shape[1])  # Upsample to match f's size
                phi_z_k = self.phi[k](z_k)
                
                # Ensure f and phi_z_k have the same shape
                if f.shape != phi_z_k.shape:
                    phi_z_k = phi_z_k[:, :f.shape[1], :]
                
                f = f - phi_z_k

        return R, vq_losses

    def decode(self, R):
        f_hat = torch.zeros_like(self.vq_layer.embedding(R[0]))
        for k in range(0, self.num_scales):
            r_k = self.vq_layer.embedding(R[k])
            z_k = self.point_cloud_upsample(f_hat, self.scale_points[k])
            phi_r_k = self.phi[k-1](r_k)
            
            # make sure z_k and phi_r_k have same shape
            if z_k.shape[1] > phi_r_k.shape[1]:
                z_k = z_k[:, :phi_r_k.shape[1], :]
            elif z_k.shape[1] < phi_r_k.shape[1]:
                phi_r_k = phi_r_k[:, :z_k.shape[1], :]
            
            f_hat = z_k + phi_r_k
        
        return self.decoder(f_hat)

    def forward(self, x):
        R, vq_losses = self.encode(x)
        reconstructed = self.decode(R)
        
        # Calculate Chamfer distance loss
        chamfer_loss = chamfer_distance(x, reconstructed)
        
        # Combine VQ losses
        vq_loss = sum(vq_losses)
        
        # Total loss is the sum of Chamfer distance and VQ losses
        total_loss = chamfer_loss + vq_loss
        
        return reconstructed, total_loss, chamfer_loss, vq_loss

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_scales):
        super(Decoder, self).__init__()
        self.num_scales = num_scales
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_scales - 1)
        ])
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1/codebook_size, 1/codebook_size)

    def forward(self, x):
        input_shape = x.shape

        # Ensure input is contiguous and flatten it
        flat_input = x.contiguous().view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Reshape encoding_indices to match input shape
        if len(input_shape) == 3:
            encoding_indices = encoding_indices.view(input_shape[0], input_shape[1])
        elif len(input_shape) == 2:
            encoding_indices = encoding_indices.view(input_shape[0])
        else:
            raise ValueError(f"Unexpected input shape: {input_shape}")
        
        # Quantize
        quantized = self.embedding(encoding_indices)
        
        # Reshape quantized to match input shape
        quantized = quantized.view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + 0.25 * e_latent_loss
        
        quantized = x + (quantized - x).detach()  # Straight-through estimator
        
        return quantized, loss, encoding_indices

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x):
        return self.conv(x)  # 移除残差连接，因为输入和输出尺寸可能不同
