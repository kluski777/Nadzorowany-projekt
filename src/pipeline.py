from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models import FeatureExtractor, Clusterizer
from models.autoencoder.architectures import ResidualConvtAutoEncoder
from models.inpainter import ConvLatentInpainter
from models.superresolution.EDSR import EDSR
from utils.device import get_device


class InferencePipeline:
    """Pipeline for latent space inpainting inference."""
    AUTOENCODER_PATH = "checkpoints/autoencoder-10k-CT-Residual-mse-latent8k-final-22301.ckpt"
    FEATURE_EXTRACTOR_PATH = "data/models/feature_extractor.pkl"
    CLUSTERIZER_PATH = "data/models/clusterizer.pkl"
    SCALER_PATH = "data/models/scaler.pkl"
    INPAINTER_DIR = "checkpoints"
    INPAINTER_PATTERN = "inpainter-cluster{cluster_id}-final.ckpt"
    COMMON_INPAINTER_PATH = "checkpoints/inpainter-common-final.ckpt"
    SUPERRESOLUTION_PATH = "checkpoints/superresolution-final.ckpt"
    INPAINTER_CFG = {"latent_channels": 128, "hidden_channels": 224, "num_blocks": 9}

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
        self.autoencoder = ResidualConvtAutoEncoder.load_from_checkpoint(self.AUTOENCODER_PATH, map_location=self.device)
        self.superresolution = EDSR.load_from_checkpoint(self.SUPERRESOLUTION_PATH, map_location=self.device)
        self.inpainter, _ = self.load_inpainter(None)
        
        for m in [self.autoencoder, self.superresolution, self.inpainter]:
            m.eval().to(self.device)
        
        self.feature_extractor = FeatureExtractor.load(self.FEATURE_EXTRACTOR_PATH)
        self.clusterizer = Clusterizer.load(self.CLUSTERIZER_PATH)
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


    def load_inpainter(self, cluster_id: int | None) -> tuple[ConvLatentInpainter, bool]:
        path = Path(self.INPAINTER_DIR) / self.INPAINTER_PATTERN.format(cluster_id=cluster_id) \
            if cluster_id is not None else Path(self.COMMON_INPAINTER_PATH)
        
        is_common = cluster_id is None or not path.exists()
        if not path.exists():
            path = Path(self.COMMON_INPAINTER_PATH)
            if not path.exists(): raise FileNotFoundError("No inpainter found.")
            print(f"  Using common inpainter for cluster {cluster_id}")

        model = ConvLatentInpainter(**self.INPAINTER_CFG, ae_path=None).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True), strict=False)
        return model.eval(), is_common

    # tutaj jeszcze Superresolution
    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.squeeze(0).cpu()
        tensor = torch.clamp(tensor, 0, 1)
        superresoluted = self.superresolution(tensor)
        array = superresoluted.permute(1, 2, 0).numpy()
        array = (array * 255).astype(np.uint8)
        
        return Image.fromarray(array)

    @torch.inference_mode()
    def inpaint(self, image: Image.Image) -> tuple[Image.Image, int]:
        input_tensor = self.preprocess(image)
        latent = self.autoencoder.encode(input_tensor)
        
        cluster_id = self.predict_cluster(latent)
        print(f"  Predicted cluster: {cluster_id}")
        
        inpainted_latent = self.inpainter(latent)
        
        output_tensor = self.autoencoder.decode(inpainted_latent)
        
        output_image = self.postprocess(output_tensor)
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        return output_image, cluster_id