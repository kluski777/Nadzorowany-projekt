import os
import argparse

import comet_ml  # noqa: F401 (import comet_ml before pytorch)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from dotenv import load_dotenv

from models.autoencoder import AutoEncoder
from data import WikiArtDataModule
from utils.config import load_config
from utils.visualize import visualize_results
from callbacks.reconstruction_logger import ReconstructionLogger

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Train AutoEncoder on WikiArt dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    seed = config["experiment"]["seed"]
    pl.seed_everything(seed, workers=True)

    data_module = WikiArtDataModule(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"],
        total_samples=config["data"]["total_samples"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"],
        data_dir=config["data"]["data_dir"],
    )

    model = AutoEncoder(
        input_channels=config["model"]["input_channels"],
        latent_channels=config["model"]["latent_channels"],
        learning_rate=config["model"]["learning_rate"],
        scheduler_patience=config["training"]["lr_scheduler_patience"],
        scheduler_factor=config["training"]["lr_scheduler_factor"],
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

    logger.log_hyperparams(config)
    logger.experiment.log_parameter("config_file", args.config)

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, recon_logger],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=config["training"]["gradient_clip_val"],
        deterministic=True,
    )

    print("Starting training...")
    trainer.fit(model, data_module)

    print("\nTraining completed!")
    print(f"Best model path: {checkpoint_callback.best_model_path}")

    print("\nGenerating visualization...")
    visualize_results(
        model, data_module, num_samples=config["experiment"]["visualization_samples"]
    )

    logger.experiment.log_image(
        "reconstruction_results.png", name="Final Reconstructions"
    )


if __name__ == "__main__":
    main()
