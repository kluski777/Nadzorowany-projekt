import pytorch_lightning as pl

from .architectures.residual_convt import ResidualConvtAutoEncoder


def get_autoencoder(architecture: str) -> pl.LightningModule:
    autoencoders = {
        "residual_convt": ResidualConvtAutoEncoder,
    }

    if architecture not in autoencoders:
        valid_types = ", ".join(f"'{k}'" for k in autoencoders.keys())
        raise ValueError(
            f"Unknown autoencoder architecture: '{architecture}'. Must be one of: {valid_types}"
        )

    return autoencoders[architecture]
