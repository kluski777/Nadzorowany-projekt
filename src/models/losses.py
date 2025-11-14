import torch
import torch.nn.functional as F


def gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create a 2D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.unsqueeze(0) * g.unsqueeze(1)


def ssim_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
) -> torch.Tensor:
    """
    Compute SSIM (Structural Similarity Index) Loss.

    Args:
        predictions: Predicted images tensor of shape (B, C, H, W)
        targets: Target images tensor of shape (B, C, H, W)
        window_size: Size of the Gaussian window (default: 11)
        C1: Constant for luminance stability (default: 0.01²)
        C2: Constant for contrast stability (default: 0.03²)

    Returns:
        SSIM loss value (1 - SSIM score)
    """
    # Create Gaussian window
    window = gaussian_kernel(window_size, sigma=1.5).to(predictions.device)
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.expand(predictions.size(1), 1, window_size, window_size)

    # Compute local means
    mu_x = F.conv2d(
        predictions, window, padding=window_size // 2, groups=predictions.size(1)
    )
    mu_y = F.conv2d(targets, window, padding=window_size // 2, groups=targets.size(1))

    # Compute local variances and covariance
    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y

    sigma_x_sq = (
        F.conv2d(
            predictions**2, window, padding=window_size // 2, groups=predictions.size(1)
        )
        - mu_x_sq
    )
    sigma_y_sq = (
        F.conv2d(targets**2, window, padding=window_size // 2, groups=targets.size(1))
        - mu_y_sq
    )
    sigma_xy = (
        F.conv2d(
            predictions * targets,
            window,
            padding=window_size // 2,
            groups=predictions.size(1),
        )
        - mu_xy
    )

    # Compute SSIM
    ssim_numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    ssim_denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_map = ssim_numerator / ssim_denominator

    return 1 - ssim_map.mean()


def ms_ssim_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor = None,
    window_size: int = 11,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
    num_scales: int = 5,
) -> torch.Tensor:
    """
    Compute MS-SSIM (Multi-Scale Structural Similarity Index) Loss.

    Args:
        predictions: Predicted images tensor of shape (B, C, H, W)
        targets: Target images tensor of shape (B, C, H, W)
        weights: Weights for each scale (default: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        window_size: Size of the Gaussian window (default: 11)
        C1: Constant for luminance stability (default: 0.01²)
        C2: Constant for contrast stability (default: 0.03²)
        num_scales: Number of scales to use (default: 5)

    Returns:
        MS-SSIM loss value (1 - MS-SSIM score)
    """
    if weights is None:
        weights = torch.tensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333][:num_scales],
            device=predictions.device,
        )
    weights = weights / weights.sum()

    # Create Gaussian window
    window = gaussian_kernel(window_size, sigma=1.5).to(predictions.device)
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.expand(predictions.size(1), 1, window_size, window_size)

    levels = []
    x = predictions
    y = targets

    for i in range(num_scales):
        # Compute statistics at current scale
        mu_x = F.conv2d(x, window, padding=window_size // 2, groups=x.size(1))
        mu_y = F.conv2d(y, window, padding=window_size // 2, groups=y.size(1))

        mu_x_sq = mu_x**2
        mu_y_sq = mu_y**2
        mu_xy = mu_x * mu_y

        sigma_x_sq = (
            F.conv2d(x**2, window, padding=window_size // 2, groups=x.size(1)) - mu_x_sq
        )
        sigma_y_sq = (
            F.conv2d(y**2, window, padding=window_size // 2, groups=y.size(1)) - mu_y_sq
        )
        sigma_xy = (
            F.conv2d(x * y, window, padding=window_size // 2, groups=x.size(1)) - mu_xy
        )

        # Compute components
        luminance = (2 * mu_xy + C1) / (mu_x_sq + mu_y_sq + C1)
        cs = (2 * sigma_xy + C2) / (sigma_x_sq + sigma_y_sq + C2)

        if i == num_scales - 1:
            # Last scale: use full SSIM (luminance * contrast_structure)
            levels.append((luminance * cs).mean())
        else:
            # Intermediate scales: use only contrast-structure
            levels.append(cs.mean())

        # Downsample for next scale
        if i < num_scales - 1:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            y = F.avg_pool2d(y, kernel_size=2, stride=2)

    # Combine scales: product of (component^weight)
    ms_ssim = torch.prod(
        torch.stack([level**weight for level, weight in zip(levels, weights)])
    )

    return 1 - ms_ssim
