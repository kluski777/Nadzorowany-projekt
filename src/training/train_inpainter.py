import os
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from dotenv import load_dotenv

from models.inpainter import ConvLatentInpainter
from data.inpainter_module import LatentInpainterDataModule

load_dotenv()


def train_inpainter(
    config: dict,
    cluster_id: Optional[int] = None,
    latent_dir: str = "data/latent_spaces",
    checkpoint_path: Optional[str] = None,
):
    """
    Train a ConvLatentInpainter model.
    
    Args:
        config: Configuration dictionary
        cluster_id: The cluster ID to train the model for. If None, trains a common
                   inpainter on all data (used as fallback when cluster-specific 
                   inpainter is not available).
        latent_dir: Directory containing the latent space npz files
        checkpoint_path: Optional path to checkpoint file (.ckpt) to resume training from
    """
    seed = config["experiment"]["seed"]
    pl.seed_everything(seed, workers=True)

    # Get inpainter-specific config
    inpainter_config = config.get("inpainter", {})
    
    # Determine model name based on cluster_id
    is_common = cluster_id is None
    model_name = "common" if is_common else f"cluster{cluster_id}"
    
    # Setup data module
    data_module = LatentInpainterDataModule(
        latent_dir=latent_dir,
        cluster_id=cluster_id,  # None = use all data
        batch_size=inpainter_config.get("batch_size", 64),
        num_workers=config["data"].get("num_workers", 4),
    )

    # Create model
    model = ConvLatentInpainter(
        latent_channels=inpainter_config.get("latent_channels", config["model"]["latent_channels"]),
        hidden_channels=inpainter_config.get("hidden_channels", 256),
        num_blocks=inpainter_config.get("num_blocks", 4),
        learning_rate=inpainter_config.get("learning_rate", 0.001),
        scheduler_patience=config["training"].get("lr_scheduler_patience", 5),
        scheduler_factor=config["training"].get("lr_scheduler_factor", 0.5),
        loss_type=inpainter_config.get("loss_type", "mse"),
    )
    print(f"\nModel architecture:\n{model}\n")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"inpainter-{model_name}-{{epoch:02d}}-{{val_loss:.6f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=config["training"].get("early_stopping_patience", 10),
        mode="min",
    )

    # Logger
    comet_api_key = os.getenv("COMET_API_KEY")
    if comet_api_key:
        logger = CometLogger(
            api_key=comet_api_key,
            project=os.getenv("COMET_PROJECT_NAME"),
            workspace=os.getenv("COMET_WORKSPACE"),
            name=f"{config['experiment']['name']}-inpainter-{model_name}",
        )
        logger.log_hyperparams({
            **config,
            "cluster_id": cluster_id,
            "is_common_inpainter": is_common,
            "latent_dir": latent_dir,
        })
    else:
        logger = None
        print("Warning: COMET_API_KEY not found. Training without Comet logging.")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=inpainter_config.get("max_epochs", config["training"]["max_epochs"]),
        accelerator="auto",
        devices="auto",
        strategy="auto",
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=config["training"].get("gradient_clip_val", 1.0),
        precision="16-mixed",
    )

    # Train
    print(f"\n{'=' * 60}")
    if is_common:
        print("Training Common Inpainter (all clusters)")
    else:
        print(f"Training Inpainter for Cluster {cluster_id}")
    print(f"{'=' * 60}\n")

    if checkpoint_path:
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.fit(model, data_module, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, data_module)

    print("\nTraining completed!")
    print(f"Best model path: {checkpoint_callback.best_model_path}")

    # Save final model
    final_model_path = f"checkpoints/inpainter-{model_name}-final.ckpt"
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # Log to Comet if available
    if logger:
        logger.experiment.log_model(
            name=f"inpainter-{model_name}-final",
            file_or_folder=final_model_path,
            metadata={
                "model_type": "ConvLatentInpainter",
                "cluster_id": cluster_id,
                "is_common_inpainter": is_common,
                "latent_channels": inpainter_config.get("latent_channels", config["model"]["latent_channels"]),
                "hidden_channels": inpainter_config.get("hidden_channels", 256),
                "final_epoch": trainer.current_epoch,
                "best_val_loss": float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score else None,
            },
        )
        print(f"Model logged to Comet ML: inpainter-{model_name}-final")
        logger.experiment.end()

    return checkpoint_callback.best_model_path
