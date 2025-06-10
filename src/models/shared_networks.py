import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim, in_channels = 1):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size = 4, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size = 4, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size = 4, stride = 2, padding = 1)

        self.fc_intermediate = nn.Linear(64 * 4 * 4, 128)
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_intermediate(x))
        return self.fc_mean(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels = 1):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 4 * 4 * 64)

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size = 4, stride = 2, padding = 1)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2, padding = 1)
        self.upconv3 = nn.ConvTranspose2d(32, 32, kernel_size = 4, stride = 2, padding = 1)
        self.upconv4 = nn.ConvTranspose2d(32, out_channels, kernel_size = 4, stride = 2, padding = 1)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), 64, 4, 4)
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))
        x_recon_logits = self.upconv4(x)
        return x_recon_logits

class Discriminator(nn.Module):
    def __init__(self, latent_dim, hidden_units=1000, num_layers=6):
        super().__init__()
        layers = []
        input_dim = latent_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_units))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            input_dim = hidden_units
        layers.append(nn.Linear(hidden_units, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)