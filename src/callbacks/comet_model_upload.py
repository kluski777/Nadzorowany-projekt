import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CometLogger


class CometModelUploadCallback(Callback):
    def __init__(self, model_name_prefix: str, comet_logger: CometLogger):
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.comet_logger = comet_logger
        self.best_val_loss = np.inf

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        current_val_loss = trainer.callback_metrics.get("val_loss")
        
        if current_val_loss is None:
            return

        if current_val_loss < self.best_val_loss:
            return
        
        self.best_val_loss = current_val_loss
        
        temp_path = f"temp_best_model_epoch_{trainer.current_epoch}.ckpt"
        trainer.save_checkpoint(temp_path)
        
        self.comet_logger.experiment.log_model(
            name=f"{self.model_name_prefix}-best-model",
            file_or_folder=temp_path,
            metadata={
                "model_type": "AutoEncoder",
                "best_val_loss": float(self.best_val_loss),
                "epoch": trainer.current_epoch,
            },
            overwrite=True,
        )
