import torch

from .base_vae import BaseVAE

class BetaVAE(BaseVAE):
    """
        A Beta-VAE model that inherits from the BaseVAE class
        It introduces the beta hyperparameter to the VAE loss function to
        encourage a more disentangled latent space
    """
    def __init__(
        self,
        latent_dim: int,
        lr: float = 1e-4,
        beta: float = 1.0,
        in_channels: int = 1,
        out_channels: int = 1,
        rec_loss: str = 'bce'
    ):
        """
            Initializes the BetaVAE model

            Args:
                latent_dim: Dimensionality of the latent space
                lr: Learning rate for the optimizer
                beta: The weight of the KL divergence term in the loss
                in_channels: Number of channels in the input image
                out_channels: Number of channels in the output image
                rec_loss: Type of reconstruction loss ('bce' or 'mse')
        """
        super().__init__(latent_dim, lr, in_channels, out_channels, rec_loss)
        self.beta = beta

    def training_step(self, batch, batch_idx):
        """
            Performs a single training step
        """
        # Unpack batch and perform forward pass
        x = batch[0]
        mean, logvar = self.encoder(x)
        z = self._reparameterize(mean, logvar)
        x_recon_logits = self.decoder(z)

        # Compute losses
        recon_loss, kl_div = self._compute_losses(x, x_recon_logits, mean, logvar)
        loss = recon_loss + self.beta * kl_div

        # Logging
        self.log_dict(
            {
                'train_loss': loss,
                'train_recon_loss': recon_loss,
                'train_kl_div': kl_div
            },
            prog_bar = True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
            Performs a single validation step
        """
        # Unpack batch and perform forward pass
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        mean, logvar = self.encoder(x)
        z = self._reparameterize(mean, logvar)
        x_recon_logits = self.decoder(z)

        # Compute losses
        recon_loss, kl_div = self._compute_losses(x, x_recon_logits, mean, logvar)
        loss = recon_loss + self.beta * kl_div

        # Logging
        self.log_dict(
            {
                'val_loss': loss,
                'val_recon_loss': recon_loss,
                'val_kl_div': kl_div
            }
        )
        return loss

    def configure_optimizers(self):
        """
            Configures the optimizer for the model
        """
        return torch.optim.Adam(
            params = self.parameters(),
            lr = self.lr_vae,
            betas = (0.9, 0.999)
        )