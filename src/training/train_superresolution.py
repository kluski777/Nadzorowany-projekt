import os
import torch
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CometLogger

from models.superresolution.EDSR import EDSR
from data.module import WikiArtDataModule
from callbacks import EpochShuffleCallback
from training.train_autoencoder import is_weights_only_checkpoint
from utils import visualize_results


def train_superresolution(config: dict, checkpoint_path: str):
    seed = config['experiment']['seed']
    pl.seed_everything(seed, workers=True)

    data_module = WikiArtDataModule(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"],
        data_dir=config["data"]["data_dir"],
        seed=seed,
    )

    model = EDSR(
        loss_type=config["superresolution"].get("loss_type", "L1"),
        hidden_size=np.ones((12), dtype=int) * 27,
        learning_rate=config['superresolution']['learning_rate'],
        scheduler_factor=config['superresolution']['scheduler_factor'],
        scheduler_patience=config['superresolution']['scheduler_patience'],
        training_image_size=config['superresolution']['training_image_size'],
        blurring=config['superresolution']['blurring']
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="superresolution-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=config["training"]["early_stopping_patience"]
    )

    logger = CometLogger(
        # api_key="L2PzW7c3YM3WqM5hNfCsloeLZ"#os.getenv('COMET_API_KEY'),
        # project_name=#os.getenv('COMET_PROJECT_NAME'),
        # workspace=os.getenv('COMET_WORKSPACE'),
        api_key="L2PzW7c3YM3WqM5hNfCsloeLZ",
        project="superresolution",
        workspace="kluski777"
    )

    epoch_shuffle_callback = EpochShuffleCallback()

    logger.log_hyperparams(config['superresolution'])

    trainer = pl.Trainer(
        max_epochs=config['superresolution']['max_epochs'],
        accelerator='auto', 
        devices='auto',
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            epoch_shuffle_callback,
        ],
        logger=logger,
        log_every_n_steps=5,
        precision='16-mixed', # zobaczymy czy w ogole zadziala z tym
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    print('Start training superresolution')
    if checkpoint_path:
        if is_weights_only_checkpoint(checkpoint_path):
            print(f'Loading pretrained weights from: {checkpoint_path}')
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt['state_dict'])
            trainer.fit(model, data_module)
        else: 
            print(f'Resuming training from checkpoint {checkpoint_path}')
            trainer.fit(model, data_module, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, data_module)

    print(f'Best superresolution model saved in {checkpoint_callback.best_model_path}')
    final_model_path = f"checkpoints/{config['superresolution']['name']}-final.ckpt"
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    print("\nGenerating visualization...")
    visualize_results(model, data_module, num_samples=config["experiment"]["visualization_samples"])

    logger.experiment.log_image("reconstruction_results_superresolution.png", name="Final Reconstructions")

    logger.experiment.log_model(
        name='SuperresolutionEDSR', # nazwa idzie i tak defaultowa
        file_or_folder=final_model_path,
        metadata={
            "model_type": "Superresolution",
            "learning_rate": config["superresolution"]["learning_rate"],
            "loss_type": config['superresolution']['loss_type'],
            "final_epoch": trainer.current_epoch,
            "best_val_loss": checkpoint_callback.best_model_score,
        },
    )
    logger.experiment.end()
