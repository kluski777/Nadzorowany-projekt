import torch
import random

#! wez to kurwa popraw to wyglada fatalito

def regular_cut(image: torch.Tensor, channels: int):
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

    if channels == 4:
        image_to_ret = torch.cat([torch.zeros(1, image_to_ret.shape[1], image_to_ret.shape[2]), image_to_ret])
        image_to_ret[0, start_h : start_h + size_h, start_w : start_w + size_w] = 1.0

    return image_to_ret


def irregular_cut(image: torch.Tensor, channels: int):
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
    
    if channels == 4:
        image_to_ret = torch.cat([torch.zeros(1, image_to_ret.shape[1], image_to_ret.shape[2]), image_to_ret])
        image_to_ret[0, h_indices, w_indices] = 1.0

    return image_to_ret


def apply_cut(image, channels: int):
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
        if channels == 4:
            image = torch.cat([torch.zeros((1, image.shape[1], image.shape[2])), image])
    elif choice < 0.66:
        image = regular_cut(image, channels)
    else:
        image = irregular_cut(image, channels)

    return image


def apply_cut_reproducible(image, seed, channels):
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

    # Use generator to determine cut type
    choice = torch.rand(1, generator=generator).item()

    if choice < 0.33:
        if channels == 4:
            image = torch.cat([torch.zeros((1, image.shape[1], image.shape[1])), image])
        return image
    elif choice < 0.66:
        # Regular cut with reproducible seed
        C, H, W = image.shape
        size_h, size_w = H // 4, W // 4

        # Use generator for random starting position
        start_h = torch.randint(0, max(1, H - size_h), (1,), generator=generator).item() if H > size_h else 0
        start_w = torch.randint(0, max(1, W - size_w), (1,), generator=generator).item() if W > size_w else 0

        image_to_ret = image.clone()
        image_to_ret[:, start_h : start_h + size_h, start_w : start_w + size_w] = 0.0
        
        if channels == 4:
            image_to_ret = torch.cat([torch.zeros(1, image_to_ret.shape[1], image_to_ret.shape[2]), image_to_ret])
            image_to_ret[0, start_h : start_h + size_h, start_w : start_w + size_w] = 1.0

        return image_to_ret
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

        if channels == 4:
            image_to_ret = torch.cat([torch.zeros(1, image_to_ret.shape[1], image_to_ret.shape[2]), image_to_ret])
            image_to_ret[0, h_indices, w_indices] = 1.0

        return image_to_ret
