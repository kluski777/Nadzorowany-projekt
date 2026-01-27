from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models import FeatureExtractor, Clusterizer
from models.autoencoder.architectures import FinalAutoEncoder2k, PixelShuffleResidualAE
from models.inpainter import ConvLatentInpainter

from utils.device import get_device


class InferencePipeline:
    """Pipeline for latent space inpainting inference."""
    # AUTOENCODER_PATH = "checkpoints/AE-latent2k.ckpt"
    AUTOENCODER_PATH = "checkpoints/AE-latent2k-PixelShuffleResidual.ckpt"
    FEATURE_EXTRACTOR_PATH = "data/models/feature_extractor.pkl"
    CLUSTERIZER_PATH = "data/models/clusterizer.pkl"
    SCALER_PATH = "data/models/scaler.pkl"
    INPAINTER_DIR = "checkpoints"
    INPAINTER_PATTERN = "inpainter-cluster{cluster_id}-final.ckpt"
    COMMON_INPAINTER_PATH = "checkpoints/inpainter-common-final.ckpt"

    def __init__(self):
        self.device = get_device()
        print(f"Using device: {self.device}")
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        self._load_models()

    def _load_models(self) -> None:
        print("Loading models...")
        
        print(f"  Loading autoencoder from: {self.AUTOENCODER_PATH}")
        self.autoencoder = FinalAutoEncoder2k.load_from_checkpoint(
            self.AUTOENCODER_PATH,
            map_location=self.device,
        )
        # self.autoencoder = PixelShuffleResidualAE.load_from_checkpoint(
        #     self.AUTOENCODER_PATH,
        #     map_location=self.device,
        # )
        self.autoencoder.eval()
        self.autoencoder.to(self.device)
        
        print(f"  Loading feature extractor from: {self.FEATURE_EXTRACTOR_PATH}")
        self.feature_extractor = FeatureExtractor.load(self.FEATURE_EXTRACTOR_PATH)
        
        print(f"  Loading clusterizer from: {self.CLUSTERIZER_PATH}")
        self.clusterizer = Clusterizer.load(self.CLUSTERIZER_PATH)
        
        print(f"  Loading scaler from: {self.SCALER_PATH}")
        self.scaler = joblib.load(self.SCALER_PATH)
        
        print("Models loaded successfully!")

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor

    def predict_cluster(self, latent: torch.Tensor) -> int:
        latent_flat = latent.cpu().numpy().reshape(1, -1)
        
        latent_components = self.feature_extractor.transform(latent_flat)
        latent_scaled = self.scaler.transform(latent_components)
        cluster_id = self.clusterizer.predict(latent_scaled)[0]
        
        return int(cluster_id)

    def load_inpainter(self, cluster_id: int) -> tuple[ConvLatentInpainter, bool]:
        inpainter_path = Path(self.INPAINTER_DIR) / self.INPAINTER_PATTERN.format(
            cluster_id=cluster_id
        )
        
        is_common = False
        if not inpainter_path.exists():
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

    # tutaj jeszcze Superresolution
    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.squeeze(0).cpu()
        tensor = torch.clamp(tensor, 0, 1)
        array = tensor.permute(1, 2, 0).numpy()
        array = (array * 255).astype(np.uint8)
        
        return Image.fromarray(array)

    @torch.inference_mode()
    def inpaint(self, image: Image.Image) -> tuple[Image.Image, int, bool]:
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
