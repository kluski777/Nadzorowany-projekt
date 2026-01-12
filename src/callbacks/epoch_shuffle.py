import pytorch_lightning as pl


class EpochShuffleCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        datamodule = trainer.datamodule
        datamodule.train_dataset.set_epoch(trainer.current_epoch)
