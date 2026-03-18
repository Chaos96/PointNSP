import torch


def punet_upsample(z: torch.Tensor, target_n: int) -> torch.Tensor:
    """
    PU-Net inspired upsampling via duplication and reshaping (Eq. 6).
    Args:
        z: (B, s_k, d) latent features at scale k
        target_n: target number of points (s_K)
    Returns:
        z_up: (B, target_n, d) upsampled features
    """
    B, s_k, d = z.shape
    if s_k == target_n:
        return z
    assert target_n % s_k == 0, f"target_n={target_n} must be divisible by s_k={s_k}"
    r = target_n // s_k
    # Duplicate: (B, s_k, d) -> (B, s_k, r, d) -> (B, s_k * r, d)
    z_up = z.unsqueeze(2).expand(B, s_k, r, d).reshape(B, s_k * r, d)
    return z_up
