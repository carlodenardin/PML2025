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

def compute_mig(model: torch.nn.Module, dataloader: DataLoader, device: str, n_samples: int = 10000) -> float:
    """
    Calculates the Mutual Information Gap (MIG) in a robust and paper-faithful way.

    MIG measures the degree of disentanglement by evaluating, for each
    ground-truth factor of variation, how much a single latent dimension
    is more informative than all others.

    Args:
        model (torch.nn.Module): The trained VAE model.
        dataloader (DataLoader): The DataLoader for the test set (containing ground-truth latents).
        device (str): The device to run computations on ('cpu', 'cuda', etc.).
        n_samples (int): The number of samples to use for the estimation.

    Returns:
        float: The calculated MIG score, normalized by the entropy of the factors.
    """
    model.eval()
    model.to(device)
    
    # --- Step 1: Collect latent samples from the model and ground-truth factors ---
    all_latents = []
    all_factors = []
    
    num_batches_processed = 0
    for batch in dataloader:
        images, factor_values = batch[0], batch[1]
        images = images.to(device)
        
        with torch.no_grad():
            mean, _ = model.encoder(images)
            all_latents.append(mean.cpu().numpy())
            all_factors.append(factor_values.cpu().numpy())
        
        num_batches_processed += 1
        # Ensure we don't iterate over the whole dataset if not needed
        if (num_batches_processed * dataloader.batch_size) >= n_samples:
            break

    latents = np.concatenate(all_latents, axis=0)[:n_samples]
    factors = np.concatenate(all_factors, axis=0)[:n_samples]
    
    n_latents = latents.shape[1]
    n_factors = factors.shape[1]
    
    # --- Step 2: Estimate the Mutual Information (MI) matrix ---
    mi_matrix = np.zeros((n_latents, n_factors))
    entropies = np.zeros(n_factors)

    for j in range(n_factors):
        # Use the raw, discrete ground-truth factors directly.
        # This is more accurate than re-discretizing them.
        factor_discrete = factors[:, j]
        
        # --- Calculate entropy H(v_j) for the normalization term ---
        # To calculate entropy, we need the probability of each unique factor value.
        _, counts = np.unique(factor_discrete, return_counts=True)
        probabilities = counts / len(factor_discrete)
        entropies[j] = entropy(probabilities, base=2)
        
        for i in range(n_latents):
            # Discretize the continuous latent dimension into 20 bins
            latent_discrete = np.digitize(latents[:, i], np.linspace(latents[:, i].min(), latents[:, i].max(), 20))
            
            # Calculate the mutual information I(z_i; v_j)
            mi_matrix[i, j] = mutual_info_score(latent_discrete, factor_discrete)

    # --- Step 3: Calculate the final MIG score ---
    # Sort the MI scores for each factor (column) in descending order
    sorted_mi = np.sort(mi_matrix, axis=0)[::-1]

    # Calculate the "gap" between the MI of the most informative and second-most informative latent
    gaps = sorted_mi[0, :] - sorted_mi[1, :]
    
    # Normalize the gaps by the factor's entropy.
    # Handle the case where entropy is 0 (for constant factors) to avoid division by zero.
    normalized_gaps = np.divide(gaps, entropies, out=np.zeros_like(gaps), where=entropies > 1e-12)
    
    # The final MIG score is the mean of the normalized gaps over all factors
    mig_score = np.mean(normalized_gaps)
    
    return float(mig_score)

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