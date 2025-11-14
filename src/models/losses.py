import torch
import torch.nn.functional as F


def ssim_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
) -> torch.Tensor:
    """
    Compute SSIM (Structural Similarity Index) Loss.

    SSIM evaluates structural similarity over pixel-wise similarity, making it ideal
    for high-quality image reconstruction. Unlike traditional losses, SSIM assesses
    luminance, contrast, and structural alignment.

    Args:
        predictions: Predicted images tensor of shape (B, C, H, W)
        targets: Target images tensor of shape (B, C, H, W)
        C1: Constant for luminance stability (default: 0.01²)
        C2: Constant for contrast stability (default: 0.03²)

    Returns:
        SSIM loss value (1 - SSIM score)
    """
    # Compute mean and variance for predictions and targets
    mu_x = F.avg_pool2d(predictions, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(targets, kernel_size=3, stride=1, padding=1)

    sigma_x = F.avg_pool2d(predictions**2, kernel_size=3, stride=1, padding=1) - mu_x**2
    sigma_y = F.avg_pool2d(targets**2, kernel_size=3, stride=1, padding=1) - mu_y**2
    sigma_xy = (
        F.avg_pool2d(predictions * targets, kernel_size=3, stride=1, padding=1)
        - mu_x * mu_y
    )

    # Compute SSIM score
    ssim_numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    ssim_score = ssim_numerator / ssim_denominator

    return 1 - ssim_score.mean()


def ms_ssim_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor = None,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
    num_scales: int = 5,
) -> torch.Tensor:
    """
    Compute MS-SSIM (Multi-Scale Structural Similarity Index) Loss.

    MS-SSIM computes SSIM at multiple scales and combines them, providing better
    assessment of image quality across different resolutions.

    Args:
        predictions: Predicted images tensor of shape (B, C, H, W)
        targets: Target images tensor of shape (B, C, H, W)
        weights: Weights for each scale (default: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        C1: Constant for luminance stability (default: 0.01²)
        C2: Constant for contrast stability (default: 0.03²)
        num_scales: Number of scales to use (default: 5)

    Returns:
        MS-SSIM loss value (1 - MS-SSIM score)
    """
    if weights is None:
        # Default weights from Wang et al. (2003)
        weights = torch.tensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=predictions.device
        )

    # Ensure weights sum to 1
    weights = weights / weights.sum()

    # Compute SSIM at each scale
    ssim_scores = []
    x = predictions
    y = targets

    for scale in range(num_scales):
        # Compute SSIM at current scale
        mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        mu_y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)

        sigma_x = F.avg_pool2d(x**2, kernel_size=3, stride=1, padding=1) - mu_x**2
        sigma_y = F.avg_pool2d(y**2, kernel_size=3, stride=1, padding=1) - mu_y**2
        sigma_xy = F.avg_pool2d(x * y, kernel_size=3, stride=1, padding=1) - mu_x * mu_y

        # For the last scale, use full SSIM formula
        if scale == num_scales - 1:
            ssim_numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
            ssim_denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
            ssim = ssim_numerator / ssim_denominator
        else:
            # For intermediate scales, use contrast and structure components only
            contrast_structure = (2 * sigma_xy + C2) / (sigma_x + sigma_y + C2)
            ssim = contrast_structure

        ssim_scores.append(ssim)

        # Downsample for next scale (except for the last iteration)
        if scale < num_scales - 1:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            y = F.avg_pool2d(y, kernel_size=2, stride=2)

    # Combine SSIM scores across scales
    ms_ssim = torch.ones_like(ssim_scores[0])
    for i, ssim in enumerate(ssim_scores):
        ms_ssim = ms_ssim * (ssim ** weights[i])

    return 1 - ms_ssim.mean()
