import torch
import torch.nn.functional as F
from typing import Callable
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def ssim_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute SSIM loss (1 - SSIM score)."""
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(predictions.device)
    ssim_score = ssim_metric(predictions, targets)
    return 1 - ssim_score


def ms_ssim_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute Multi-Scale SSIM loss (1 - MS-SSIM score)."""
    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(
        predictions.device
    )
    ms_ssim_score = ms_ssim_metric(predictions, targets)
    return 1 - ms_ssim_score


def lpips_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute LPIPS perceptual loss using VGG features."""
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(
        predictions.device
    )
    return lpips_metric(predictions, targets)


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
    }
    
    if loss_type not in loss_functions:
        valid_types = ", ".join(f"'{k}'" for k in loss_functions.keys())
        raise ValueError(
            f"Unknown loss_type: '{loss_type}'. Must be one of: {valid_types}"
        )
    
    return loss_functions[loss_type]
