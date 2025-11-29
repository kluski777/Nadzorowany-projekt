import torch
import torch.nn.functional as F
from typing import Callable, Dict, Any
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def ssim_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    _cache: Dict[str, Any] = {},
) -> torch.Tensor:
    """Compute SSIM loss (1 - SSIM score). Metric is cached in _cache parameter."""
    if "metric" not in _cache:
        _cache["metric"] = StructuralSimilarityIndexMeasure(data_range=1.0)

    metric = _cache["metric"]
    if metric.device != predictions.device:
        metric = metric.to(predictions.device)
        _cache["metric"] = metric

    ssim_score = metric(predictions, targets)
    return 1 - ssim_score


def ms_ssim_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    _cache: Dict[str, Any] = {},
) -> torch.Tensor:
    """Compute Multi-Scale SSIM loss (1 - MS-SSIM score). Metric is cached in _cache parameter."""
    if "metric" not in _cache:
        _cache["metric"] = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

    metric = _cache["metric"]
    if metric.device != predictions.device:
        metric = metric.to(predictions.device)
        _cache["metric"] = metric

    ms_ssim_score = metric(predictions, targets)
    return 1 - ms_ssim_score


def lpips_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    _cache: Dict[str, Any] = {},
) -> torch.Tensor:
    """Compute LPIPS perceptual loss using VGG features. Metric is cached in _cache parameter."""
    if "metric" not in _cache:
        _cache["metric"] = LearnedPerceptualImagePatchSimilarity(net_type="vgg")

    metric = _cache["metric"]
    if metric.device != predictions.device:
        metric = metric.to(predictions.device)
        _cache["metric"] = metric

    return metric(predictions, targets)


def mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute Mean Squared Error loss."""
    return F.mse_loss(predictions, targets)


def mae_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute Mean Absolute Error (L1) loss."""
    return F.l1_loss(predictions, targets)


def huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute Huber loss (robust to outliers)."""
    return F.huber_loss(predictions, targets)


def bce_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute Binary Cross Entropy loss."""
    return F.binary_cross_entropy(predictions, targets)


def combined_mse_bce_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mse_weight: float = 1.0,
    bce_weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute combined loss as weighted sum of MSE and BCE losses.
    
    Args:
        predictions: Predicted values
        targets: Target values
        mse_weight: Multiplicative factor for MSE loss. Defaults to 1.0.
        bce_weight: Multiplicative factor for BCE loss. Defaults to 1.0.
    
    Returns:
        Combined loss: mse_weight * MSE + bce_weight * BCE
    """
    mse = F.mse_loss(predictions, targets)
    bce = F.binary_cross_entropy(predictions, targets)
    return mse_weight * mse + bce_weight * bce


def get_loss_function(loss_type: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Get loss function by name.

    Args:
        loss_type: One of 'ssim', 'ms_ssim', 'lpips', 'mse', 'mae', 'huber'

    Returns:
        Loss function callable

    Raises:
        ValueError: If loss_type is not recognized
    """
    loss_functions = {
        "ssim": ssim_loss,
        "ms_ssim": ms_ssim_loss,
        "lpips": lpips_loss,
        "mse": mse_loss,
        "mae": mae_loss,
        "huber": huber_loss,
        "bce": bce_loss,
        "combined_mse_bce": combined_mse_bce_loss,
    }

    if loss_type not in loss_functions:
        valid_types = ", ".join(f"'{k}'" for k in loss_functions.keys())
        raise ValueError(
            f"Unknown loss_type: '{loss_type}'. Must be one of: {valid_types}"
        )

    return loss_functions[loss_type]
