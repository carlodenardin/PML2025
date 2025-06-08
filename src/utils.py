import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import random
from sklearn.metrics import mutual_info_score
from config import *

def get_random_images(dataloader, n_images, device):
    random.seed(SEED)
    first_batch = next(iter(dataloader))
    if isinstance(first_batch, (list, tuple)):
        images = first_batch[0]
    else:
        images = first_batch
    batch_size = images.shape[0]
    indices = random.sample(range(batch_size), n_images)
    selected_images = images[indices]
    return selected_images.to(device)

def save_reconstructions(
    model,
    dataloader,
    device: str,
    result_dir: str,
    output_filename: str = "reconstructions.png"
):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents = True, exist_ok = True)
    save_path = result_dir / output_filename

    selected_images = get_random_images(dataloader, 5, device)

    with torch.no_grad():
        mean, logvar = model.encoder(selected_images)
        z = model._reparameterize(mean, logvar)
        reconstructed_logits = model.decoder(z)
        reconstructed_images = torch.sigmoid(reconstructed_logits)

    fig, axes = plt.subplots(len(selected_images), 2, figsize = (4, len(selected_images) * 2), squeeze = False)
    for i in range(len(selected_images)):
        axes[i, 0].imshow(selected_images[i].cpu().squeeze().numpy(), cmap='gray')
        axes[i, 0].set_title(f"Original {i + 1}")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(reconstructed_images[i].cpu().squeeze().numpy(), cmap='gray')
        axes[i, 1].set_title(f"Reconstructed {i + 1}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def compute_mig(model, dataloader, n_samples=1000, device='cpu'):
    """Calcola il Mutual Information Gap (MIG) per valutare il disentanglement."""
    model.eval()
    model.to(device)
    latents, factors = [], []

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            images, factor_values = batch
        else:
            images = batch
            factor_values = None
        images = images.to(device)
        with torch.no_grad():
            mean, _ = model.encoder(images)
            latents.append(mean.cpu().numpy())
            if factor_values is not None:
                factors.append(factor_values.cpu().numpy())
        if len(latents) * dataloader.batch_size >= n_samples:
            break

    latents = np.concatenate(latents, axis=0)[:n_samples]
    factors = np.concatenate(factors, axis=0)[:n_samples] if factors else None

    if factors is None:
        raise ValueError("Latent factors required for MIG computation")

    n_latents, n_factors = latents.shape[1], factors.shape[1]
    mi_matrix = np.zeros((n_latents, n_factors))
    for i in range(n_latents):
        for j in range(n_factors):
            latent_vals = np.digitize(latents[:, i], np.linspace(latents[:, i].min(), latents[:, i].max(), 20))
            factor_vals = np.digitize(factors[:, j], np.linspace(factors[:, j].min(), factors[:, j].max(), 20))
            mi_matrix[i, j] = mutual_info_score(latent_vals, factor_vals)

    sorted_mi = np.sort(mi_matrix, axis=0)[::-1]
    mig = (sorted_mi[0] - sorted_mi[1]).mean() / sorted_mi[0].mean() if sorted_mi[0].mean() > 0 else 0
    return mig

def get_accelerator():
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except AttributeError:
        pass
    return "cpu"

from pytorch_lightning.callbacks import Callback

class MIG(Callback):
    """
    Callback per calcolare il MIG score alla fine di ogni epoca di validazione.
    """
    def on_validation_epoch_end(self, trainer, pl_module):
        """Chiamato alla fine dell'epoca di validazione."""
        val_dataloader = trainer.datamodule.val_dataloader()
        device = pl_module.device
        mig_score = compute_mig(pl_module, val_dataloader, device=device)
        pl_module.log('val_mig', mig_score, prog_bar=True)

def get_folder_name(checkpoint_path: str):
    path = Path(checkpoint_path)
    return path.parent.name

def get_seed(checkpoint_path: str):
    folder_name = get_folder_name(checkpoint_path)
    match = re.search(r"seed_(\d+)", folder_name)
    
    if match:
        seed_str = match.group(1)
        return folder_name, int(seed_str)