from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .shared_networks import Decoder, Encoder

class BaseVAE(pl.LightningModule, ABC):
    """
        An abstract base class for Variational Autoencoders (VAEs)
        It defines the core architecture and loss computations, while leaving
        the specific training and validation steps to be implemented by subclasses
    """
    def __init__(
        self,
        latent_dim: int,
        lr_vae: float,
        in_channels: int = 1,
        out_channels: int = 1,
        rec_loss: str = 'bce'
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.latent_dim = latent_dim
        self.lr_vae = lr_vae
        self.rec_loss = rec_loss
        
        self.encoder = Encoder(latent_dim, in_channels)
        self.decoder = Decoder(latent_dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Defines the forward pass of the VAE, which corresponds to the encoder
        """
        return self.encoder(x)

    def _reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
            Applies the reparameterization trick to sample from the latent space to for backpropagation
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def _compute_losses(self, x: torch.Tensor, x_recon_logits: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
            Computes the two main components of the VAE loss: reconstruction and KL divergence.
        """
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim = 1).mean()

        # Reconstruction Loss
        if self.rec_loss == 'bce':
            recon_loss = F.binary_cross_entropy_with_logits(x_recon_logits, x, reduction = 'none').sum(dim = [1, 2, 3]).mean()
        else:
            recon_loss = F.mse_loss(torch.sigmoid(x_recon_logits), x, reduction = 'none').sum(dim = [1, 2, 3]).mean()
        
        return recon_loss, kl_div

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass