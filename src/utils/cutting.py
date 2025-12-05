import torch
import random
import numpy as np

def regular_cut(image: torch.Tensor):
    """
    Apply a regular rectangular cut to a single image.

    Args:
        image: Tensor of shape (C, H, W)

    Returns:
        Modified image tensor with cut applied (pixels set to 0)
    """
    _, H, W = image.shape
    size_h, size_w = H // 4, W // 4

    # Random starting position
    start_h = random.randint(0, H - size_h - 1) if H > size_h else 0
    start_w = random.randint(0, W - size_w - 1) if W > size_w else 0

    image_to_ret = image.clone()
    image_to_ret[:, start_h : start_h + size_h, start_w : start_w + size_w] = 0.0

    mask = np.zeros((1, H, W))
    mask[:, start_h:start_h+size_h, start_w:start_w+size_w] = 1.0

    return image_to_ret, mask


def irregular_cut(image: torch.Tensor):
    """
    Apply an irregular cut to a single image by removing scattered pixels.

    Args:
        image: Tensor of shape (C, H, W)

    Returns:
        Modified image tensor with cut applied (pixels set to 0)
    """
    _, H, W = image.shape
    cut_points_num = (H * W) // 8  # 1/8 of pixels will be removed

    # Random center point
    center_h = random.uniform(0, H)
    center_w = random.uniform(0, W)

    # Generate offsets around center
    scaling = H // 16
    offsets_h = torch.randn(cut_points_num) * scaling
    offsets_w = torch.randn(cut_points_num) * scaling

    # Calculate indices to cut
    h_indices = ((offsets_h + center_h).long() % H).clamp(0, H - 1)
    w_indices = ((offsets_w + center_w).long() % W).clamp(0, W - 1)

    image_to_ret = image.clone()
    image_to_ret[:, h_indices, w_indices] = 0.0

    mask = torch.zeros(1, H, W)
    mask[:, h_indices, w_indices] = 1.0
    return image_to_ret, mask


def apply_cut(image):
    """
    Randomly apply a cut to an image (33% no cut, 33% regular, 33% irregular).

    Args:
        image: Tensor of shape (C, H, W)
        channels - number of channels to return (with mask or not)

    Returns:
        Modified image tensor
    """

    choice = random.random()

    if choice < 0.33:
        mask = np.zeros((1, image.shape[-2], image.shape[-1]))
        image = image
    elif choice < 0.66:
        image, mask = regular_cut(image)
    else:
        image, mask = irregular_cut(image)

    return image, mask


def apply_cut_reproducible(image, seed):
    """
    Apply a cut to an image deterministically based on seed.
    Uses 33% probability for each cut type (none, regular, irregular).

    Args:
        image: Tensor of shape (C, H, W)
        seed: Integer seed for reproducibility

    Returns:
        Modified image tensor
    """
    # Create a generator with the seed for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)

    _, H, W = image.shape

    # Use generator to determine cut type
    choice = torch.rand(1, generator=generator).item()
    mask = torch.zeros((1, H, W))

    if choice < 0.33:
        image_to_ret = image
    elif choice < 0.66:
        # Regular cut with reproducible seed
        size_h, size_w = H // 4, W // 4

        # Use generator for random starting position
        start_h = torch.randint(0, max(1, H - size_h), (1,), generator=generator).item() if H > size_h else 0
        start_w = torch.randint(0, max(1, W - size_w), (1,), generator=generator).item() if W > size_w else 0

        image_to_ret = image.clone()
        image_to_ret[:, start_h : start_h + size_h, start_w : start_w + size_w] = 0.0
        
        mask[:, start_h: start_h + size_h, start_w : start_w + size_w] = 1.0
    else:
        # Irregular cut with reproducible seed
        C, H, W = image.shape
        cut_points_num = (H * W) // 8

        # Use generator for center point
        center_h = torch.rand(1, generator=generator).item() * H
        center_w = torch.rand(1, generator=generator).item() * W

        # Generate offsets around center
        scaling = H // 16
        offsets_h = torch.randn(cut_points_num, generator=generator) * scaling
        offsets_w = torch.randn(cut_points_num, generator=generator) * scaling

        # Calculate indices to cut
        h_indices = ((offsets_h + center_h).long() % H).clamp(0, H - 1)
        w_indices = ((offsets_w + center_w).long() % W).clamp(0, W - 1)

        image_to_ret = image.clone()
        image_to_ret[:, h_indices, w_indices] = 0.0
        mask[:, h_indices, w_indices] = 1.0
        
    return image_to_ret, mask
