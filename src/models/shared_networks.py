import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder network as described in Appendix A, Table 1 of the paper for 2D Shapes.
    Input: (batch_size, 1, 64, 64)
    Output: (batch_size, latent_dim), (batch_size, latent_dim) for mean and log_var
    """
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)  # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1) # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # 16x16 -> 8x8
        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1) # 8x8 -> 4x4

        self.fc_intermediate = nn.Linear(64 * 4 * 4, 128)
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc_intermediate(x))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

class Decoder(nn.Module):
    """
    Decoder network as described in Appendix A, Table 1 of the paper for 2D Shapes.
    Input: (batch_size, latent_dim)
    Output: (batch_size, 1, 64, 64) - logits for Bernoulli distribution
    """
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 4 * 4 * 64)

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1) # 4x4 -> 8x8
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # 8x8 -> 16x16
        self.upconv3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1) # 16x16 -> 32x32
        self.upconv4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)  # 32x32 -> 64x64

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
    """
    Discriminator network for FactorVAE as described in Appendix A.
    "6 layer MLP discriminator with 1000 hidden units per layer and leaky ReLU (lReLU) non-linearity,
    that outputs 2 logits in all FactorVAE experiments."
    Here, we implement it to output a single logit for P(z is from q(z) vs q_bar(z)).
    Input: (batch_size, latent_dim)
    Output: (batch_size, 1) - logits
    """
    def __init__(self, latent_dim, hidden_units=1000, num_layers=6):
        super(Discriminator, self).__init__()
        
        layers = []
        input_dim = latent_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_units))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            input_dim = hidden_units
        
        layers.append(nn.Linear(hidden_units, 1)) # Single logit output
        
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)