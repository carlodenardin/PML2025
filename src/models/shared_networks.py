import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
        A convolutional encoder for the VAE
        Maps an input image to the parameters (mean and log-variance)
        of a latent distribution
    """
    def __init__(self, latent_dim: int, in_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size = 4, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size = 4, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size = 4, stride = 2, padding = 1)

        # Fully-connected layers
        self.fc_intermediate = nn.Linear(64 * 4 * 4, 128)
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
            Performs the forward pass of the encoder

            Args:
                x: Input tensor of shape (N, C, H, W).

            Returns:
                A tuple containing the mean and log-variance of the latent distribution.
        """
        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Flatten and pass through fully-connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_intermediate(x))
        
        # Compute mean and log-variance
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        
        return mean, logvar

class Decoder(nn.Module):
    """
        A convolutional decoder for the VAE
        Maps a latent code z back to a reconstructed image.
    """
    def __init__(self, latent_dim: int, out_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim

        # Fully-connected layers
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 4 * 4 * 64)

        # Transposed convolutional layers
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size = 4, stride = 2, padding = 1)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2, padding = 1)
        self.upconv3 = nn.ConvTranspose2d(32, 32, kernel_size = 4, stride = 2, padding = 1)
        self.upconv4 = nn.ConvTranspose2d(32, out_channels, kernel_size = 4, stride = 2, padding = 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
            Performs the forward pass of the decoder

            Args:
                z: Latent code tensor of shape (N, latent_dim)

            Returns:
                The reconstructed image logits tensor of shape (N, C, H, W)
        """
        # Pass through fully-connected layers and reshape
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), 64, 4, 4)
        
        # Pass through transposed convolutional layers
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))
        
        # Final output layer (logits)
        x_recon_logits = self.upconv4(x)
        
        return x_recon_logits

class Discriminator(nn.Module):
    """
        A Multi-Layer Perceptron (MLP) discriminator for Factor-VAE
        It takes a latent code z and outputs a single logit for classification
    """
    def __init__(self, latent_dim: int, hidden_units: int = 1000, num_layers: int = 6):
        super().__init__()
        
        layers = []
        input_dim = latent_dim
        
        # Build the hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_units))
            layers.append(nn.LeakyReLU(0.2, inplace = True))
            input_dim = hidden_units
            
        # Output layer
        layers.append(nn.Linear(hidden_units, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
            Performs the forward pass of the discriminator.

            Args:
                z: Latent code tensor of shape (N, latent_dim).

            Returns:
                A single logit value for each sample in the batch.
        """
        return self.model(z)