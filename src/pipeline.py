"""
Inference pipeline for latent space inpainting.

Handles model loading and the full inference flow:
masked image -> encode -> cluster -> inpaint -> decode -> reconstructed image
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models import FeatureExtractor, Clusterizer
from models.autoencoder.architectures import FinalAutoEncoder2k
from models.inpainter import ConvLatentInpainter


class InferencePipeline:
    """
    Pipeline for latent space inpainting inference.
    
    Loads autoencoder, feature extractor, clusterizer, and inpainters
    to process masked images and return reconstructed images.
    """

    # Hardcoded model paths (from README.md conventions)
    AUTOENCODER_PATH = "checkpoints/AE-latent2k.ckpt"
    FEATURE_EXTRACTOR_PATH = "data/models/feature_extractor.pkl"
    CLUSTERIZER_PATH = "data/models/clusterizer.pkl"
    SCALER_PATH = "data/models/scaler.pkl"
    INPAINTER_DIR = "checkpoints"
    INPAINTER_PATTERN = "inpainter-cluster{cluster_id}-final.ckpt"
    COMMON_INPAINTER_PATH = "checkpoints/inpainter-common-final.ckpt"

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the inference pipeline.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # Load models
        self._load_models()

    def _load_models(self) -> None:
        """Load all required models."""
        print("Loading models...")
        
        # Load autoencoder
        print(f"  Loading autoencoder from: {self.AUTOENCODER_PATH}")
        self.autoencoder = FinalAutoEncoder2k.load_from_checkpoint(
            self.AUTOENCODER_PATH,
            map_location=self.device,
        )
        self.autoencoder.eval()
        self.autoencoder.to(self.device)
        
        # Load feature extractor
        print(f"  Loading feature extractor from: {self.FEATURE_EXTRACTOR_PATH}")
        self.feature_extractor = FeatureExtractor.load(self.FEATURE_EXTRACTOR_PATH)
        
        # Load clusterizer
        print(f"  Loading clusterizer from: {self.CLUSTERIZER_PATH}")
        self.clusterizer = Clusterizer.load(self.CLUSTERIZER_PATH)
        
        # Load scaler
        print(f"  Loading scaler from: {self.SCALER_PATH}")
        self.scaler = joblib.load(self.SCALER_PATH)
        
        print("Models loaded successfully!")

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image: PIL Image (any size, RGB or RGBA)
            
        Returns:
            Tensor of shape (1, 3, 256, 256)
        """
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Apply transforms and add batch dimension
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor

    def predict_cluster(self, latent: torch.Tensor) -> int:
        """
        Predict the cluster ID for a latent representation.
        
        Args:
            latent: Latent tensor of shape (1, channels, H, W)
            
        Returns:
            Cluster ID (integer)
        """
        latent_flat = latent.cpu().numpy().reshape(1, -1)
        
        latent_components = self.feature_extractor.transform(latent_flat)
        latent_scaled = self.scaler.transform(latent_components)
        cluster_id = self.clusterizer.predict(latent_scaled)[0]
        
        return int(cluster_id)

    def load_inpainter(self, cluster_id: int) -> tuple[ConvLatentInpainter, bool]:
        """
        Load the inpainter model for a specific cluster.
        
        Falls back to the common inpainter if cluster-specific one is not available.
        Note: Inpainters are loaded fresh each time (not cached) to save memory.
        
        Args:
            cluster_id: The cluster ID
            
        Returns:
            Tuple of (ConvLatentInpainter model, is_common: bool)
        """
        # Try cluster-specific inpainter first
        inpainter_path = Path(self.INPAINTER_DIR) / self.INPAINTER_PATTERN.format(
            cluster_id=cluster_id
        )
        
        is_common = False
        if not inpainter_path.exists():
            # Fall back to common inpainter
            common_path = Path(self.COMMON_INPAINTER_PATH)
            if not common_path.exists():
                raise FileNotFoundError(
                    f"No inpainter found for cluster {cluster_id} ({inpainter_path}) "
                    f"and no common inpainter available ({common_path}). "
                    "Train either a cluster-specific or common inpainter first."
                )
            inpainter_path = common_path
            is_common = True
            print(f"  Cluster {cluster_id} inpainter not found, using common inpainter")
        
        print(f"  Loading inpainter from: {inpainter_path}")
        inpainter = ConvLatentInpainter.load_from_checkpoint(
            str(inpainter_path),
            map_location=self.device,
        )
        inpainter.eval()
        inpainter.to(self.device)
        
        return inpainter, is_common

    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert output tensor back to PIL Image.
        
        Args:
            tensor: Tensor of shape (1, 3, H, W) with values in [0, 1]
            
        Returns:
            PIL Image
        """
        tensor = tensor.squeeze(0).cpu()
        tensor = torch.clamp(tensor, 0, 1)
        array = tensor.permute(1, 2, 0).numpy()
        array = (array * 255).astype(np.uint8)
        
        return Image.fromarray(array)

    @torch.inference_mode()
    def inpaint(self, image: Image.Image) -> tuple[Image.Image, int, bool]:
        """
        Run the full inpainting pipeline.
        
        Args:
            image: Input masked image (PIL Image)
            
        Returns:
            Tuple of (reconstructed PIL Image, cluster_id, used_common_inpainter)
        """
        input_tensor = self.preprocess(image)
        latent = self.autoencoder.encode(input_tensor)
        
        cluster_id = self.predict_cluster(latent)
        print(f"  Predicted cluster: {cluster_id}")
        
        inpainter, is_common = self.load_inpainter(cluster_id)
        inpainted_latent = inpainter(latent)
        
        output_tensor = self.autoencoder.decode(inpainted_latent)
        
        output_image = self.postprocess(output_tensor)
        
        del inpainter
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        return output_image, cluster_id, is_common
