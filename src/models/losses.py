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
    if "metric" not in _cache:
        _cache["metric"] = LearnedPerceptualImagePatchSimilarity(net_type="vgg")

    metric = _cache["metric"]
    if metric.device != predictions.device:
        metric = metric.to(predictions.device)
        _cache["metric"] = metric

    return metric(predictions, targets)


def mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predictions, targets)


def mae_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(predictions, targets)


def huber_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(predictions, targets)


def bce_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(predictions, targets)


def combined_l1_bce_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    l1_weight: float = 1.0,
    bce_weight: float = 0.1,
) -> torch.Tensor:
    l1 = F.l1_loss(predictions, targets)
    bce = F.binary_cross_entropy(predictions, targets)
    return l1_weight * l1 + bce_weight * bce


def combined_mse_bce_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mse_weight: float = 1.0,
    bce_weight: float = 1.0,
) -> torch.Tensor:
    mse = F.mse_loss(predictions, targets)
    bce = F.binary_cross_entropy(predictions, targets)
    return mse_weight * mse + bce_weight * bce


def vae_bce_kl_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + kl_loss
    
    return {
        'loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss
    }


def get_loss_function(loss_type: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    loss_functions = {
        "ssim": ssim_loss,
        "ms_ssim": ms_ssim_loss,
        "lpips": lpips_loss,
        "mse": mse_loss,
        "mae": mae_loss,
        "huber": huber_loss,
        "bce": bce_loss,
        "combined_mse_bce": combined_mse_bce_loss,
        "combined_l1_bce_loss": combined_l1_bce_loss,
        "vae_bce_kl": vae_bce_kl_loss,
    }

    if loss_type not in loss_functions:
        valid_types = ", ".join(f"'{k}'" for k in loss_functions.keys())
        raise ValueError(f"Unknown loss_type: '{loss_type}'. Must be one of: {valid_types}")

    return loss_functions[loss_type]
