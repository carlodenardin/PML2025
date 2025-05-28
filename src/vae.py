import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class DisentangleVAE(pl.LightningModule):
    
    def __init__(self, img_size, nb_channels, z_dim, beta = 1.0, learning_rate = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.img_size = img_size
        self.nb_channels = nb_channels
        self.z_dim = z_dim
        self.beta = beta
        self.learning_rate = learning_rate

        self.encoder_layers = nn.Sequential(
            nn.Conv2d(self.nb_channels, 64, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.encoder_output_dim = 512 * (self.img_size // 16) * (self.img_size // 16)
        self.fc_mu = nn.Linear(self.encoder_output_dim, self.z_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_dim, self.z_dim)

        self.decoder_input_channels = 512
        self.decoder_initial_spatial_size = self.img_size // 16
        self.fc_decoder = nn.Linear(self.z_dim, self.decoder_input_channels * self.decoder_initial_spatial_size * self.decoder_initial_spatial_size)

        self.decoder_layer = nn.Sequential(
            nn.ConvTranspose2d(self.decoder_input_channels, 256, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.nb_channels, kernel_size = 4, stride = 2, padding = 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_layers(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.fc_decoder(z)
        x = x.view(x.size(0), self.decoder_input_channels, self.decoder_initial_spatial_size, self.decoder_initial_spatial_size)
        x = self.decoder_layer(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar
    
    def loss_function(self, x, x_reconstructed, mu, logvar):
        recon_loss = F.binary_cross_entropy(x_reconstructed, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_divergence
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_reconstructed, mu, logvar = self(x)
        loss = self.loss_function(x, x_reconstructed, mu, logvar)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_reconstructed, mu, logvar = self(x)
        loss = self.loss_function(x, x_reconstructed, mu, logvar)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss