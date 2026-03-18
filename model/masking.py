import torch
import torch.nn.functional as F


def build_blockwise_causal_mask(scale_points: list, device: torch.device = None) -> torch.Tensor:
    """Build block-wise causal attention mask.

    Inter-scale: causal (lower-triangular across blocks)
    Intra-scale: fully unmasked (bidirectional within each block)

    Args:
        scale_points: list [s_1, s_2, ..., s_K]
        device: torch device
    Returns:
        mask: (T, T) binary mask where T = sum(scale_points). 1=attend, 0=mask.
    """
    total = sum(scale_points)
    mask = torch.zeros(total, total, device=device)

    # Compute block offsets
    offsets = [0]
    for s in scale_points:
        offsets.append(offsets[-1] + s)

    K = len(scale_points)
    for i in range(K):       # row block (query scale)
        for j in range(K):   # col block (key scale)
            if j <= i:        # causal: scale i can attend to scale j if j <= i
                mask[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]] = 1.0

    return mask


def build_position_aware_soft_mask(P_k: torch.Tensor, W_p: torch.Tensor) -> torch.Tensor:
    """Position-aware soft masking matrix M_k^p (Eq. 8).

    M_k^p = Softmax((P_k @ W_p) @ (P_k @ W_p)^T)

    Args:
        P_k: (B, s_k, d) positional embeddings for scale k
        W_p: (d, d) learnable projection matrix
    Returns:
        mask: (B, s_k, s_k) soft mask with values in (0, 1)
    """
    projected = P_k @ W_p  # (B, s_k, d)
    logits = projected @ projected.transpose(-1, -2)  # (B, s_k, s_k)
    mask = F.softmax(logits, dim=-1)  # row-wise softmax
    return mask
