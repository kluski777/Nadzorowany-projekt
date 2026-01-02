import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from dotenv import load_dotenv

from models.autoencoder.architectures.bottleneck_variants import (
    BottleneckAE4k,
    BottleneckAE2k,
    BottleneckAE1k,
)
from data import WikiArtDataModule
from utils import visualize_results
from callbacks import ReconstructionLogger, EpochShuffleCallback, CometModelUploadCallback

load_dotenv()


def train_bottleneck(config: dict):
    """
    Train the specified bottleneck variant.

    Args:
        config: Configuration dictionary with keys:
            - model.variant: "4k", "2k", or "1k"
            - model.pretrained_checkpoint: path to previous stage checkpoint
            - model.learning_rate: learning rate
            - model.loss_type: loss function type
            - data.*: data configuration
            - training.*: training configuration
            - experiment.*: experiment configuration
    """
    seed = config["experiment"]["seed"]
    pl.seed_everything(seed, workers=True)

    cutting_config = config.get("cutting", {})
    cutting_seed = cutting_config.get("seed")
    if cutting_seed is None:
        cutting_seed = seed

    # Data setup
    data_module = WikiArtDataModule(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"],
        data_dir=config["data"]["data_dir"],
        seed=seed,
        splits_dir=config["data"]["splits_dir"],
        enable_cutting=cutting_config.get("enable", False),
        cutting_mode_train=cutting_config.get("mode_train", "random"),
        cutting_mode_val=cutting_config.get("mode_val", "reproducible"),
        cutting_mode_test=cutting_config.get("mode_test", "reproducible"),
        cutting_seed=cutting_seed,
    )

    variant = config["model"]["variant"]
    checkpoint = config["model"]["pretrained_checkpoint"]
    learning_rate = config["model"]["learning_rate"]
    loss_type = config["model"].get("loss_type", "mse")
    scheduler_patience = config["training"]["lr_scheduler_patience"]
    scheduler_factor = config["training"]["lr_scheduler_factor"]

    # Create model based on variant
    if variant == "4k":
        model = BottleneckAE4k(
            base_checkpoint=checkpoint,
            learning_rate=learning_rate,
            scheduler_patience=scheduler_patience,
            scheduler_factor=scheduler_factor,
            loss_type=loss_type,
        )
        latent_dim = 4096
    elif variant == "2k":
        model = BottleneckAE2k(
            ae4k_checkpoint=checkpoint,
            learning_rate=learning_rate,
            scheduler_patience=scheduler_patience,
            scheduler_factor=scheduler_factor,
            loss_type=loss_type,
        )
        latent_dim = 2048
    elif variant == "1k":
        model = BottleneckAE1k(
            ae2k_checkpoint=checkpoint,
            learning_rate=learning_rate,
            scheduler_patience=scheduler_patience,
            scheduler_factor=scheduler_factor,
            loss_type=loss_type,
        )
        latent_dim = 1024
    else:
        raise ValueError(f"Unknown variant: {variant}. Must be '4k', '2k', or '1k'")

    experiment_name = config["experiment"]["name"]

    print(f"\n{'='*60}")
    print(f"Training BottleneckAE{variant} (latent dim: {latent_dim})")
    print(f"Previous checkpoint: {checkpoint}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")

    # Checkpoint directory
    checkpoint_dir = f"checkpoints/bottleneck_{variant}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"bottleneck-{variant}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=config["training"]["early_stopping_patience"],
        mode="min",
    )

    logger = CometLogger(
        api_key=os.getenv("COMET_API_KEY"),
        project=os.getenv("COMET_PROJECT_NAME"),
        workspace=os.getenv("COMET_WORKSPACE"),
        name=experiment_name,
    )

    recon_logger = ReconstructionLogger(
        log_every_n_epochs=config["experiment"]["recon_log_every_n_epochs"],
        num_samples=config["experiment"]["visualization_samples"],
    )

    epoch_shuffle_callback = EpochShuffleCallback()

    comet_upload_callback = CometModelUploadCallback(
        model_name_prefix=experiment_name,
        comet_logger=logger,
    )

    config_to_log = config.copy()
    config_to_log["bottleneck"] = {
        "variant": variant,
        "latent_dim": latent_dim,
        "previous_checkpoint": checkpoint,
    }
    logger.log_hyperparams(config_to_log)

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="auto",
        devices="auto",
        strategy="auto",
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            recon_logger,
            epoch_shuffle_callback,
            comet_upload_callback,
        ],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=config["training"]["gradient_clip_val"],
        precision="16-mixed",
    )

    print("Starting training...")
    trainer.fit(model, data_module)

    print("\nTraining completed!")
    print(f"Best model path: {checkpoint_callback.best_model_path}")

    # Save final model
    final_model_path = f"{checkpoint_dir}/{experiment_name}-final.ckpt"
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # Visualization
    print("\nGenerating visualization...")
    visualize_results(model, data_module, num_samples=config["experiment"]["visualization_samples"])
    logger.experiment.log_image("reconstruction_results.png", name="Final Reconstructions")

    # Log model to Comet
    logger.experiment.log_model(
        name=f"{experiment_name}-final-model",
        file_or_folder=final_model_path,
        metadata={
            "model_type": f"BottleneckAE{variant}",
            "latent_dim": latent_dim,
            "learning_rate": learning_rate,
            "final_epoch": trainer.current_epoch,
            "best_val_loss": float(checkpoint_callback.best_model_score)
            if checkpoint_callback.best_model_score
            else None,
        },
    )
    print(f"Model logged to Comet ML: {experiment_name}-final-model")

    logger.experiment.end()

    return model, checkpoint_callback.best_model_path

