import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import random
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from config import *
from pytorch_lightning.callbacks import Callback

def get_folder_name(checkpoint_path: str):
    path = Path(checkpoint_path)
    return path.parent.name

def get_seed(checkpoint_path: str):
    folder_name = get_folder_name(checkpoint_path)
    match = re.search(r"seed_(\d+)", folder_name)
    
    if match:
        seed_str = match.group(1)
        return folder_name, int(seed_str)

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

def get_accelerator():
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except AttributeError:
        pass
    return "cpu"

def save_reconstructions(
    model,
    dataloader: DataLoader,
    device: str,
    result_dir: str,
    output_filename: str = "reconstructions.png"
):
    """
    Saves a plot of original vs. reconstructed images.
    This function now handles both grayscale and RGB images.
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    save_path = result_dir / output_filename

    # Get a small batch of random images
    selected_images = get_random_images(dataloader, 5, device)
    if selected_images is None:
        print("Could not generate reconstructions: no images found.")
        return

    # Generate reconstructions
    with torch.no_grad():
        mean, logvar = model.encoder(selected_images)
        # Using the mean for a deterministic reconstruction
        reconstructed_logits = model.decoder(mean)
        reconstructed_images = torch.sigmoid(reconstructed_logits)

    # Create plot
    fig, axes = plt.subplots(len(selected_images), 2, figsize=(4, len(selected_images) * 2), squeeze=False)
    
    for i in range(len(selected_images)):
        original_img = selected_images[i].cpu()
        reconstructed_img = reconstructed_images[i].cpu()

        # --- KEY CHANGE: Handle both Grayscale and RGB images ---
        if original_img.shape[0] == 1:
            # Grayscale image (C, H, W) -> (H, W)
            original_np = original_img.squeeze().numpy()
            reconstructed_np = reconstructed_img.squeeze().numpy()
            cmap = 'gray'
        else:
            # RGB image (C, H, W) -> (H, W, C)
            original_np = original_img.permute(1, 2, 0).numpy()
            reconstructed_np = reconstructed_img.permute(1, 2, 0).numpy()
            cmap = None # Let matplotlib infer colormap for RGB

        # Plot original
        axes[i, 0].imshow(original_np, cmap=cmap)
        axes[i, 0].set_title(f"Original {i + 1}")
        axes[i, 0].axis('off')
        
        # Plot reconstructed
        axes[i, 1].imshow(reconstructed_np, cmap=cmap)
        axes[i, 1].set_title(f"Reconstructed {i + 1}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Reconstructions saved to {save_path}")

def compute_mig(model, dataloader, device, n_samples = 10000):
    model.eval().to(device)
    latents, factors = [], []

    # Raccogli latenti e fattori
    for images, factor_vals in dataloader:
        images = images.to(device)
        with torch.no_grad():
            mean, _ = model.encoder(images)
        latents.append(mean.cpu().numpy())
        factors.append(factor_vals.cpu().numpy())
        if len(latents) * dataloader.batch_size >= n_samples:
            break

    latents = np.concatenate(latents, axis=0)[:n_samples]
    factors = np.concatenate(factors, axis=0)[:n_samples]

    n_latents, n_factors = latents.shape[1], factors.shape[1]
    mi_matrix = np.zeros((n_latents, n_factors))
    entropies = np.zeros(n_factors)

    for j in range(n_factors):
        unique_vals = len(np.unique(factors[:, j]))
        bins = unique_vals if unique_vals <= 4 else 20
        factor_vals = factors[:, j].astype(int) if unique_vals <= 4 else np.digitize(
            factors[:, j], np.linspace(factors[:, j].min(), factors[:, j].max(), bins))
        
        counts = np.bincount(factor_vals) / len(factor_vals)
        entropies[j] = entropy(counts, base=2)

        for i in range(n_latents):
            latent_vals = np.digitize(latents[:, i], np.linspace(latents[:, i].min(), latents[:, i].max(), 20))
            mi_matrix[i, j] = mutual_info_score(latent_vals, factor_vals)

    sorted_mi = np.sort(mi_matrix, axis=0)[::-1]
    mig_scores = (sorted_mi[0] - sorted_mi[1]) / np.maximum(entropies, 1e-10)
    return np.mean(mig_scores[np.isfinite(mig_scores)])

"""
def compute_mig(model, dataloader, n_samples, device):
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
"""

class MIG(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        val_dataloader = trainer.datamodule.val_dataloader()
        device = pl_module.device
        mig_score = compute_mig(pl_module, val_dataloader, device = device)
        pl_module.log('val_mig', mig_score, prog_bar = True)