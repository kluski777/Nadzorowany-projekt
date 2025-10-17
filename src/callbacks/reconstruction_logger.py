import torch
from pytorch_lightning.callbacks import Callback
import torchvision.utils as vutils


class ReconstructionLogger(Callback):
    """Logs reconstruction examples to Comet every N epochs."""

    def __init__(self, log_every_n_epochs: int = 5, num_samples: int = 8):
        """
        Args:
            log_every_n_epochs: How often to log images
            num_samples: Number of image pairs to log (original + reconstruction)
        """
        super().__init__()

        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self.sample_batch = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Capture first validation batch for logging."""
        if batch_idx == 0 and self.sample_batch is None:
            self.sample_batch = {
                "image": batch["image"][: self.num_samples].detach().cpu()
            }

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log reconstructions every N epochs."""
        
        if self.sample_batch is None:
            return

        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        images = self.sample_batch["image"].to(pl_module.device)

        pl_module.eval()
        with torch.inference_mode():
            reconstructed = pl_module(images)
            if isinstance(reconstructed, tuple):
                reconstructed = reconstructed[0]

        pl_module.train()

        images = images.cpu()
        reconstructed = reconstructed.cpu()

        comparison = torch.stack([images, reconstructed], dim=1)
        comparison = comparison.view(-1, *images.shape[1:])

        grid = vutils.make_grid(
            comparison,
            nrow=2,
            normalize=True,
            value_range=(0, 1),
            padding=2,
        )

        if trainer.logger is not None:
            trainer.logger.experiment.log_image(
                grid.permute(1, 2, 0).numpy(),
                name="reconstructions",
                step=trainer.current_epoch,
            )
