import pytorch_lightning as pl

from models.autoencoder.architectures import ResidualConvtAutoEncoder, ResK1UpsampleAutoEncoder, ResNet18AutoEncoder, PixelShuffleAE


def get_autoencoder(architecture: str) -> pl.LightningModule:
    autoencoders = {
        "res_convt": ResidualConvtAutoEncoder,
        "res_k_1_upsample": ResK1UpsampleAutoEncoder,
        "resnet18_ae": ResNet18AutoEncoder,
        "pixelshuffle_ae": PixelShuffleAE,
    }

    if architecture not in autoencoders:
        valid_types = ", ".join(f"'{k}'" for k in autoencoders.keys())
        raise ValueError(f"Unknown autoencoder architecture: '{architecture}'. Must be one of: {valid_types}")

    return autoencoders[architecture]
