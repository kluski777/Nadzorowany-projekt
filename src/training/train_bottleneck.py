import os
import argparse

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train bottleneck autoencoder variants")
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["4k", "2k", "1k"],
        help="Which bottleneck variant to train (4k, 2k, or 1k)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the previous stage checkpoint (base 8k for 4k, 4k for 2k, 2k for 1k)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=150,
        help="Maximum epochs (default: 150)",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        help="Loss type (default: mse)",
    )
    return parser.parse_args()


def train_bottleneck(args):
    """Train the specified bottleneck variant."""
    pl.seed_everything(42, workers=True)

    # Data setup
    data_module = WikiArtDataModule(
        batch_size=args.batch_size,
        num_workers=3,
        image_size=256,
        data_dir="./data",
        seed=42,
        splits_dir="splits",
        enable_cutting=True,
        cutting_mode_train="random",
        cutting_mode_val="reproducible",
        cutting_mode_test="reproducible",
        cutting_seed=42,
    )

    # Create model based on variant
    if args.variant == "4k":
        model = BottleneckAE4k(
            base_checkpoint=args.checkpoint,
            learning_rate=args.learning_rate,
            scheduler_patience=5,
            scheduler_factor=0.5,
            loss_type=args.loss_type,
        )
        latent_dim = 4096
    elif args.variant == "2k":
        model = BottleneckAE2k(
            ae4k_checkpoint=args.checkpoint,
            learning_rate=args.learning_rate,
            scheduler_patience=5,
            scheduler_factor=0.5,
            loss_type=args.loss_type,
        )
        latent_dim = 2048
    else:  # 1k
        model = BottleneckAE1k(
            ae2k_checkpoint=args.checkpoint,
            learning_rate=args.learning_rate,
            scheduler_patience=5,
            scheduler_factor=0.5,
            loss_type=args.loss_type,
        )
        latent_dim = 1024

    experiment_name = f"bottleneck-{args.variant}"

    print(f"\n{'='*60}")
    print(f"Training BottleneckAE{args.variant} (latent dim: {latent_dim})")
    print(f"Previous checkpoint: {args.checkpoint}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"{'='*60}\n")

    # Checkpoint directory
    checkpoint_dir = f"checkpoints/bottleneck_{args.variant}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"bottleneck-{args.variant}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )

    logger = CometLogger(
        api_key=os.getenv("COMET_API_KEY"),
        project=os.getenv("COMET_PROJECT_NAME"),
        workspace=os.getenv("COMET_WORKSPACE"),
        name=experiment_name,
    )

    recon_logger = ReconstructionLogger(
        log_every_n_epochs=5,
        num_samples=8,
    )

    epoch_shuffle_callback = EpochShuffleCallback()

    comet_upload_callback = CometModelUploadCallback(
        model_name_prefix=experiment_name,
        comet_logger=logger,
    )

    logger.log_hyperparams({
        "variant": args.variant,
        "latent_dim": latent_dim,
        "previous_checkpoint": args.checkpoint,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "loss_type": args.loss_type,
    })

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
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
        gradient_clip_val=1.0,
        precision="16-mixed",
    )

    print("Starting training...")
    trainer.fit(model, data_module)

    print("\nTraining completed!")
    print(f"Best model path: {checkpoint_callback.best_model_path}")

    # Save final model
    final_model_path = f"{checkpoint_dir}/bottleneck-{args.variant}-final.ckpt"
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # Visualization
    print("\nGenerating visualization...")
    visualize_results(model, data_module, num_samples=8)
    logger.experiment.log_image("reconstruction_results.png", name="Final Reconstructions")

    # Log model to Comet
    logger.experiment.log_model(
        name=f"{experiment_name}-final-model",
        file_or_folder=final_model_path,
        metadata={
            "model_type": f"BottleneckAE{args.variant}",
            "latent_dim": latent_dim,
            "learning_rate": args.learning_rate,
            "final_epoch": trainer.current_epoch,
            "best_val_loss": float(checkpoint_callback.best_model_score)
            if checkpoint_callback.best_model_score
            else None,
        },
    )
    print(f"Model logged to Comet ML: {experiment_name}-final-model")

    logger.experiment.end()

    return model, checkpoint_callback.best_model_path


if __name__ == "__main__":
    args = parse_args()
    train_bottleneck(args)

