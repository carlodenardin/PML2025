import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .shared_networks import Encoder, Decoder

class VAE(pl.LightningModule):
    def __init__(self, latent_dim: int, lr: float = 1e-4, beta: float = 1.0):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.lr = lr
        self.beta = beta # Weight for KL divergence, beta=1 for standard VAE

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        return mean, logvar

    def _reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def training_step(self, batch, batch_idx):
        x = batch
        mean, logvar = self.encoder(x)
        z = self._reparameterize(mean, logvar)
        x_recon_logits = self.decoder(z)

        # Reconstruction loss (Binary Cross Entropy with Logits)
        # Sum over pixels, then mean over batch
        recon_loss = F.binary_cross_entropy_with_logits(x_recon_logits, x, reduction='none').sum(dim=[1,2,3]).mean()

        # KL divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # Sum over latent dimensions, then mean over batch
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()

        loss = recon_loss + self.beta * kl_div

        self.log_dict({
            'train_loss': loss,
            'train_recon_loss': recon_loss,
            'train_kl_div': kl_div
        }, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        mean, logvar = self.encoder(x)
        z = self._reparameterize(mean, logvar)
        x_recon_logits = self.decoder(z)
        
        recon_loss = F.binary_cross_entropy_with_logits(x_recon_logits, x, reduction='none').sum(dim=[1,2,3]).mean()
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
        loss = recon_loss + self.beta * kl_div

        self.log_dict({
            'val_loss': loss,
            'val_recon_loss': recon_loss,
            'val_kl_div': kl_div
        })
        return loss

    def configure_optimizers(self):
        # Adam optimizer params from paper Appendix A [cite: 255]
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        return optimizer