from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .shared_networks import Encoder, Decoder

class BaseVAE(pl.LightningModule, ABC):
    def __init__(self, latent_dim, lr_vae):
        super().__init__()
        self.save_hyperparameters()
        
        self.latent_dim = latent_dim
        self.lr_vae = lr_vae
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        return self.encoder(x)

    def _reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def _compute_losses(self, x, x_recon_logits, mean, logvar):
        recon_loss = F.binary_cross_entropy_with_logits(x_recon_logits, x, reduction='none').sum(dim=[1, 2, 3]).mean()
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
        return recon_loss, kl_div

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass