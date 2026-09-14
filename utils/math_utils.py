import torch

# @torch.compile

# @torch.compile

# @torch.compile

# Strange synchronization here if using torch.compile

# @torch.compile

# @torch.compile


# @torch.compile
def affine_padding(c2w: torch.Tensor):
    # Already padded
    if c2w.shape[-2] == 4:
        return c2w
    # Batch agnostic padding
    sh = c2w.shape
    pad0 = c2w.new_zeros(sh[:-2] + (1, 3))  # B, 1, 3
    pad1 = c2w.new_ones(sh[:-2] + (1, 1))  # B, 1, 1
    pad = torch.cat([pad0, pad1], dim=-1)  # B, 1, 4
    c2w = torch.cat([c2w, pad], dim=-2)  # B, 4, 4
    return c2w


# @torch.compile
def ixt_padding(K: torch.Tensor):
    # Already padded
    if K.shape[-2] == 4:
        return K
    # Batch agnostic padding
    sh = K.shape
    canvas = K.new_zeros(sh[:-2] + (4, 4))  # B, 4, 4
    canvas[..., :3, :3] = K
    canvas[..., 3, 3] = 1.0
    return canvas


# @torch.compile
def affine_inverse(A: torch.Tensor):
    R = A[..., :3, :3]  # ..., 3, 3
    T = A[..., :3, 3:]  # ..., 3, 1
    P = A[..., 3:, :]  # ..., 1, 4
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)


# @torch.compile
def ixt_inverse(K: torch.Tensor):
    device = K.device
    sh = K.shape[:-2]  # B, ..., 3, 3
    inv_K = torch.eye(3, device=device)  # 3, 3
    for s in sh[::-1]:
        inv_K = inv_K[None].expand((s,) + inv_K.shape)
    inv_K = inv_K.contiguous()
    inv_K[..., 0, 0] = 1.0 / K[..., 0, 0]
    inv_K[..., 1, 1] = 1.0 / K[..., 1, 1]
    inv_K[..., 0, 2] = -K[..., 0, 2] / K[..., 0, 0]
    inv_K[..., 1, 2] = -K[..., 1, 2] / K[..., 1, 1]
    return inv_K


# these works with an extra batch dimension
# Batched inverse of lower triangular matrices

# @torch.compile

# @torch.compile

# @torch.compile

# @torch.compile

# @torch.compile

# @torch.compile

# @torch.compile
