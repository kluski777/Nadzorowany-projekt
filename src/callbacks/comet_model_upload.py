import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger


class CometModelUploadCallback(Callback):

    def __init__(self, model_name_prefix: str, comet_logger: CometLogger):
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.comet_logger = comet_logger
        self._checkpoint_callback = None
        self._last_uploaded_score = None

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                self._checkpoint_callback = callback
                break

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._checkpoint_callback is None:
            return

        current_val_loss = trainer.callback_metrics.get("val_loss")
        best_score = self._checkpoint_callback.best_model_score

        if (
            current_val_loss is not None
            and best_score is not None
            and current_val_loss == best_score
            and self._last_uploaded_score != best_score
        ):
            self.comet_logger.experiment.log_model(
                name=f"{self.model_name_prefix}-best-model",
                file_or_folder=self._checkpoint_callback.best_model_path,
                metadata={
                    "model_type": "AutoEncoder",
                    "best_val_loss": float(best_score),
                    "epoch": trainer.current_epoch,
                },
                overwrite=True,
            )
            self._last_uploaded_score = best_score
