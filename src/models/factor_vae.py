import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .base_vae import BaseVAE
from .shared_networks import Discriminator

class FactorVAE(BaseVAE):
    """
        A Factor-VAE model that inherits from the BaseVAE class
        It uses a discriminator in an adversarial setup to approximate the
        Total Correlation (TC) term, encouraging a disentangled latent space
    """
    def __init__(
        self,
        latent_dim: int,
        lr_vae: float = 1e-4,
        lr_disc: float = 1e-4,
        gamma: float = 10.0,
        disc_hidden_units: int = 1000,
        disc_layers: int = 6,
        in_channels: int = 1,
        out_channels: int = 1,
        rec_loss: str = 'bce'
    ):
        """
            Initializes the FactorVAE model

            Args:
                latent_dim: Dimensionality of the latent space
                lr_vae: Learning rate for the VAE (encoder/decoder)
                lr_disc: Learning rate for the discriminator
                gamma: The weight of the Total Correlation (TC) term
                disc_hidden_units: Number of hidden units in the discriminator
                disc_layers: Number of layers in the discriminator
                in_channels: Number of channels in the input image
                out_channels: Number of channels in the output image
                rec_loss: Type of reconstruction loss ('bce' or 'mse')
        """
        super().__init__(latent_dim, lr_vae, in_channels, out_channels, rec_loss)
        
        self.lr_disc = lr_disc
        self.gamma = gamma
        self.automatic_optimization = False
        
        self.discriminator = Discriminator(
            input_dim = latent_dim,
            hidden_units = disc_hidden_units,
            num_layers = disc_layers
        )

    def _permute_dims(self, z: torch.Tensor) -> torch.Tensor:
        """
            Permutes the dimensions of the latent code `z` across the batch.
            This breaks correlations between dimensions and creates the "fake" samples
            for the discriminator.
        """
        B, D = z.shape
        permuted_z = torch.zeros_like(z)
        for i in range(D):
            # Permute indices for each dimension independently
            idx = torch.randperm(B, device = z.device)
            permuted_z[:, i] = z[idx, i]
        return permuted_z

    def training_step(self, batch, batch_idx):
        """
            Performs a single training step with manual optimization for both the
            VAE and the discriminator.
        """
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        opt_vae, opt_disc = self.optimizers()

        # --- VAE/Generator Training Step ---
        # Forward pass through VAE
        mean, logvar = self.encoder(x)
        z_samples = self._reparameterize(mean, logvar)
        x_recon_logits = self.decoder(z_samples)

        # Calculate VAE losses
        recon_loss, kl_div_vae = self._compute_losses(x, x_recon_logits, mean, logvar)
        
        # Adversarial loss (Total Correlation term proxy)
        D_z_logits = self.discriminator(z_samples)
        tc_loss_vae = (D_z_logits[:, 0] - D_z_logits[:, 1]).mean()

        # Combine for total VAE loss
        vae_loss = recon_loss + kl_div_vae + self.gamma * tc_loss_vae

        # Optimize VAE (encoder and decoder)
        opt_vae.zero_grad()
        self.manual_backward(vae_loss)
        opt_vae.step()

        # --- Discriminator Training Step ---
        # Create real and fake samples (detach to avoid backprop into encoder)
        z_real_samples = z_samples.detach()
        z_permuted_samples = self._permute_dims(z_real_samples)

        # Get discriminator predictions
        D_real_logits = self.discriminator(z_real_samples)
        D_fake_logits = self.discriminator(z_permuted_samples)

        # Calculate discriminator loss (binary cross-entropy)
        loss_d_real = F.binary_cross_entropy_with_logits(D_real_logits, torch.ones_like(D_real_logits))
        loss_d_fake = F.binary_cross_entropy_with_logits(D_fake_logits, torch.zeros_like(D_fake_logits))
        disc_loss = 0.5 * (loss_d_real + loss_d_fake)

        # Optimize Discriminator
        opt_disc.zero_grad()
        self.manual_backward(disc_loss)
        opt_disc.step()

        # --- Logging ---
        self.log_dict({
            'train_vae_loss': vae_loss,
            'train_recon_loss': recon_loss,
            'train_kl_div_vae': kl_div_vae,
            'train_tc_loss_vae_term': tc_loss_vae,
            'train_disc_loss': disc_loss,
            'D_real_pred_mean': torch.sigmoid(D_real_logits[:, 0]).mean(),
            'D_fake_pred_mean': torch.sigmoid(D_fake_logits[:, 0]).mean(),
        }, prog_bar = True)
        return vae_loss

    def validation_step(self, batch, batch_idx):
        """
            Performs a single validation step
        """
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        
        # VAE Loss Calculation
        mean, logvar = self.encoder(x)
        z = self._reparameterize(mean, logvar)
        x_recon_logits = self.decoder(z)
        recon_loss, kl_div_vae = self._compute_losses(x, x_recon_logits, mean, logvar)
        D_z_logits = self.discriminator(z)
        tc_loss_val_term = (D_z_logits[:, 0] - D_z_logits[:, 1]).mean()
        val_loss = recon_loss + kl_div_vae + self.gamma * tc_loss_val_term

        # Discriminator Loss Calculation
        z_permuted = self._permute_dims(z.detach())
        D_fake_logits = self.discriminator(z_permuted)
        loss_d_real_val = F.binary_cross_entropy_with_logits(D_z_logits, torch.ones_like(D_z_logits))
        loss_d_fake_val = F.binary_cross_entropy_with_logits(D_fake_logits, torch.zeros_like(D_fake_logits))
        val_disc_loss = 0.5 * (loss_d_real_val + loss_d_fake_val)

        # Logging
        self.log_dict({
            'val_loss': val_loss,
            'val_recon_loss': recon_loss,
            'val_kl_div_vae': kl_div_vae,
            'val_disc_loss': val_disc_loss
        })
        return val_loss

    def configure_optimizers(self):
        """
            Configures two separate optimizers: one for the VAE (encoder/decoder)
            and one for the discriminator
        """
        # Optimizer for Encoder and Decoder
        opt_vae = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr = self.lr_vae, 
            betas = (0.9, 0.999)
        )
        
        # Optimizer for Discriminator
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr = self.lr_disc, 
            betas = (0.5, 0.9)
        )