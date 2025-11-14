import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from dotenv import load_dotenv

from models.autoencoder import AutoEncoder
from data import WikiArtDataModule
from utils import load_config, visualize_results
from callbacks import ReconstructionLogger, EpochShuffleCallback

load_dotenv()


def train_autoencoder(config_path: str, checkpoint_path: str = None):
    """
    Train AutoEncoder on WikiArt dataset.

    Args:
        config_path: Path to configuration YAML file
        checkpoint_path: Optional path to checkpoint file (.ckpt) to resume training from
    """
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    seed = config["experiment"]["seed"]
    pl.seed_everything(seed, workers=True)

    cutting_config = config.get("cutting", {})
    cutting_seed = cutting_config.get("seed")
    if cutting_seed is None:
        cutting_seed = seed

    data_module = WikiArtDataModule(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"],
        data_dir=config["data"]["data_dir"],
        shuffle_buffer_size=config["data"]["shuffle_buffer_size"],
        seed=seed,
        splits_dir=config["data"]["splits_dir"],
        enable_cutting=cutting_config.get("enable", False),
        cutting_mode_train=cutting_config.get("mode_train", "random"),
        cutting_mode_val=cutting_config.get("mode_val", "reproducible"),
        cutting_mode_test=cutting_config.get("mode_test", "reproducible"),
        cutting_seed=cutting_seed,
    )

    model = AutoEncoder(
        input_channels=config["model"]["input_channels"],
        latent_channels=config["model"]["latent_channels"],
        learning_rate=config["model"]["learning_rate"],
        scheduler_patience=config["training"]["lr_scheduler_patience"],
        scheduler_factor=config["training"]["lr_scheduler_factor"],
        loss_type=config["model"].get("loss_type", "ssim"),
    )
    print(f"\nModel architecture:\n{model}\n")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="autoencoder-{epoch:02d}-{val_loss:.4f}",
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
        name=config["experiment"]["name"],
    )

    recon_logger = ReconstructionLogger(
        log_every_n_epochs=config["experiment"]["recon_log_every_n_epochs"],
        num_samples=config["experiment"]["visualization_samples"],
    )

    epoch_shuffle_callback = EpochShuffleCallback()

    logger.log_hyperparams(config)
    logger.experiment.log_parameter("config_file", config_path)

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="auto",
        devices=-1,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            recon_logger,
            epoch_shuffle_callback,
        ],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=config["training"]["gradient_clip_val"],
        deterministic=True,
    )

    print("Starting training...")
    if checkpoint_path:
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.fit(model, data_module, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, data_module)

    print("\nTraining completed!")
    print(f"Best model path: {checkpoint_callback.best_model_path}")

    final_model_path = f"checkpoints/{config['experiment']['name']}-final.ckpt"
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    print("\nGenerating visualization...")
    visualize_results(
        model, data_module, num_samples=config["experiment"]["visualization_samples"]
    )

    logger.experiment.log_image(
        "reconstruction_results.png", name="Final Reconstructions"
    )

    logger.experiment.log_model(
        name=f"{config['experiment']['name']}-final-model",
        file_or_folder=final_model_path,
        metadata={
            "model_type": "AutoEncoder",
            "input_channels": config["model"]["input_channels"],
            "latent_channels": config["model"]["latent_channels"],
            "learning_rate": config["model"]["learning_rate"],
            "final_epoch": trainer.current_epoch,
            "best_val_loss": checkpoint_callback.best_model_score,
        },
    )
    print(f"Model logged to Comet ML: {config['experiment']['name']}-final-model")

    logger.experiment.end()
