import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .shared_networks import Encoder, Decoder, Discriminator

class FactorVAE(pl.LightningModule):
    def __init__(self, latent_dim: int, 
                 lr_vae: float = 1e-4, lr_disc: float = 1e-4, 
                 gamma: float = 10.0, disc_hidden_units: int = 1000, disc_layers: int = 6):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False # Manual optimization for VAE and Discriminator

        self.latent_dim = latent_dim
        self.lr_vae = lr_vae
        self.lr_disc = lr_disc
        self.gamma = gamma # Weight for TC term

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.discriminator = Discriminator(latent_dim, hidden_units=disc_hidden_units, num_layers=disc_layers)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        return mean, logvar

    def _reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def _permute_dims(self, z):
        """
        Permutes dimensions of z across the batch for q_bar(z) sampling.
        Algorithm 1 from the paper[cite: 63].
        Input: z (batch_size, latent_dim)
        Output: z_permuted (batch_size, latent_dim)
        """
        assert z.ndim == 2
        B, D = z.shape
        permuted_z = torch.zeros_like(z)
        for i in range(D):
            # Create a random permutation of batch indices for dimension i
            idx = torch.randperm(B, device=z.device)
            permuted_z[:, i] = z[idx, i]
        return permuted_z

    def training_step(self, batch, batch_idx):
        x = batch
        opt_vae, opt_disc = self.optimizers()

        # VAE (Generator) training phase
        # VAE wants to:
        # 1. Reconstruct x well
        # 2. Keep q(z|x) close to p(z) (unit Gaussian prior)
        # 3. Make q(z) (aggregate posterior) factorized, i.e., minimize TC(z)
        #    TC(z) approx E_q(z)[log D(z) - log(1-D(z))]
        #    VAE wants to make D(z) small, so it minimizes (log D(z) - log(1-D(z)))
        
        mean, logvar = self.encoder(x)
        z_samples = self._reparameterize(mean, logvar) # Samples from q(z|x)
        x_recon_logits = self.decoder(z_samples)

        # VAE Reconstruction loss
        recon_loss = F.binary_cross_entropy_with_logits(x_recon_logits, x, reduction='none').sum(dim=[1,2,3]).mean()

        # VAE KL divergence (q(z|x) || p(z))
        kl_div_vae = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
        
        # Total Correlation (TC) loss term for VAE
        # D_z_logits are logits from discriminator for z_samples ~ q(z)
        # VAE wants to make D(z_samples) small (i.e., make D think z_samples are from q_bar)
        # Loss term for VAE is gamma * (log D(z) - log(1-D(z)))
        # (log D(z) - log(1-D(z))) is simply the logit if D(z) = sigmoid(logit)
        D_z_logits = self.discriminator(z_samples) # Output of discriminator for "real" q(z) samples
        
        # The paper's objective (Eq 2) is L_VAE_base - gamma * TC. To minimize, it's -L_VAE_base + gamma*TC.
        # TC is approx E_q(z)[log(D(z)/(1-D(z)))].
        # So VAE loss = -Recon + KL + gamma * (log D(z) - log(1-D(z)))
        # This term encourages D(z) to be small.
        tc_loss_vae = (D_z_logits).mean() # Equivalent to log(D(z)/(1-D(z))) if D(z) is sigmoid(logit)

        vae_loss = recon_loss + kl_div_vae + self.gamma * tc_loss_vae
        
        opt_vae.zero_grad()
        self.manual_backward(vae_loss) # Gradient for encoder and decoder
        opt_vae.step()

        # Discriminator training phase
        # Discriminator wants to:
        # 1. Classify z_samples from q(z) as "real" (label 1)
        # 2. Classify z_permuted_samples from q_bar(z) as "fake" (label 0)
        
        # Detach z_samples so that gradients don't flow back to encoder for this step
        z_real_samples = z_samples.detach() 
        z_permuted_samples = self._permute_dims(z_real_samples)

        # D_real_logits: discriminator output for samples from q(z)
        # D_fake_logits: discriminator output for samples from q_bar(z) (permuted)
        D_real_logits = self.discriminator(z_real_samples)
        D_fake_logits = self.discriminator(z_permuted_samples)

        # Discriminator loss: BCEWithLogitsLoss
        # Wants D_real_logits to be high (predict 1) and D_fake_logits to be low (predict 0)
        loss_d_real = F.binary_cross_entropy_with_logits(D_real_logits, torch.ones_like(D_real_logits))
        loss_d_fake = F.binary_cross_entropy_with_logits(D_fake_logits, torch.zeros_like(D_fake_logits))
        disc_loss = 0.5 * (loss_d_real + loss_d_fake)

        opt_disc.zero_grad()
        self.manual_backward(disc_loss) # Gradient for discriminator
        opt_disc.step()

        self.log_dict({
            'train_vae_loss': vae_loss,
            'train_recon_loss': recon_loss,
            'train_kl_div_vae': kl_div_vae,
            'train_tc_loss_vae_term': tc_loss_vae, # This is E_q(z)[logit(D(z))]
            'train_disc_loss': disc_loss,
            'D_real_pred_mean': torch.sigmoid(D_real_logits).mean(), # Avg prob D thinks real is real
            'D_fake_pred_mean': torch.sigmoid(D_fake_logits).mean(), # Avg prob D thinks fake is real
        }, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        x = batch
        mean, logvar = self.encoder(x)
        z = self._reparameterize(mean, logvar)
        x_recon_logits = self.decoder(z)
        
        recon_loss = F.binary_cross_entropy_with_logits(x_recon_logits, x, reduction='none').sum(dim=[1,2,3]).mean()
        kl_div_vae = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
        
        # For validation, we are primarily interested in VAE's performance
        # The TC loss for VAE depends on D, which is also evolving.
        # We can log the VAE parts and D's accuracy if desired.
        D_z_logits = self.discriminator(z)
        tc_loss_val_term = D_z_logits.mean()
        val_vae_loss = recon_loss + kl_div_vae + self.gamma * tc_loss_val_term

        z_permuted = self._permute_dims(z.detach())
        D_fake_logits = self.discriminator(z_permuted)
        loss_d_real_val = F.binary_cross_entropy_with_logits(D_z_logits, torch.ones_like(D_z_logits))
        loss_d_fake_val = F.binary_cross_entropy_with_logits(D_fake_logits, torch.zeros_like(D_fake_logits))
        val_disc_loss = 0.5 * (loss_d_real_val + loss_d_fake_val)

        self.log_dict({
            'val_vae_loss': val_vae_loss,
            'val_recon_loss': recon_loss,
            'val_kl_div_vae': kl_div_vae,
            'val_disc_loss': val_disc_loss
        })
        return val_vae_loss


    def configure_optimizers(self):
        # Adam optimizer params from paper Appendix A [cite: 255, 256]
        opt_vae = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                   lr=self.lr_vae, betas=(0.9, 0.999))
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=self.lr_disc, betas=(0.5, 0.9)) # Note: different betas for discriminator
        return opt_vae, opt_disc