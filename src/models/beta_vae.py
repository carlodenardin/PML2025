import torch
from .base_vae import BaseVAE

class BetaVAE(BaseVAE):
    def __init__(
            self,
            latent_dim: int,
            lr: float = 1e-4,
            beta: float = 1.0,
            in_channels: int = 1,
            out_channels: int = 1,
            rec_loss: str = 'bce'
        ):
        super().__init__(latent_dim, lr, in_channels, out_channels, rec_loss)
        self.beta = beta

    def training_step(self, batch, batch_idx):
        x = batch[0]
        mean, logvar = self.encoder(x)
        z = self._reparameterize(mean, logvar)
        x_recon_logits = self.decoder(z)
        recon_loss, kl_div = self._compute_losses(x, x_recon_logits, mean, logvar)
        loss = recon_loss + self.beta * kl_div
        self.log_dict({'train_loss': loss, 'train_recon_loss': recon_loss, 'train_kl_div': kl_div}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        mean, logvar = self.encoder(x)
        z = self._reparameterize(mean, logvar)
        x_recon_logits = self.decoder(z)
        recon_loss, kl_div = self._compute_losses(x, x_recon_logits, mean, logvar)
        loss = recon_loss + self.beta * kl_div
        self.log_dict({'val_loss': loss, 'val_recon_loss': recon_loss, 'val_kl_div': kl_div})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr_vae, betas=(0.9, 0.999))