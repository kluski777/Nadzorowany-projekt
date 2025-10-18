import pytorch_lightning as pl


class EpochShuffleCallback(pl.Callback):
    """Callback to update the epoch number in streaming datasets for proper shuffling."""
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Update the epoch in the training dataset for epoch-dependent shuffling."""
        datamodule = trainer.datamodule
        if hasattr(datamodule, 'train_dataset') and hasattr(datamodule.train_dataset, 'set_epoch'):
            datamodule.train_dataset.set_epoch(trainer.current_epoch)
            print(f"Epoch {trainer.current_epoch}: Updated shuffle seed")
