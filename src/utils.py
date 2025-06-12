import torch
import matplotlib.pyplot as plt
import numpy as np
import re
import random
from pathlib import Path
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from pytorch_lightning.callbacks import Callback

def get_folder_name(checkpoint_path: str) -> str:
    return Path(checkpoint_path).parent.name

def get_seed(checkpoint_path: str) -> tuple[str, int] | tuple[str, None]:
    folder_name = get_folder_name(checkpoint_path)
    match = re.search(r"seed_(\d+)", folder_name)
    
    if match:
        seed_str = match.group(1)
        return folder_name, int(seed_str)
    return folder_name, None

def get_random_images(dataloader: torch.utils.data.DataLoader, n_images: int, device: str) -> torch.Tensor | None:
    if not dataloader:
        return None

    random.seed(random.randint(0, 10000))
    
    try:
        first_batch = next(iter(dataloader))
    except StopIteration:
        print("Dataloader is empty, cannot get random images.")
        return None

    images = first_batch[0] if isinstance(first_batch, (list, tuple)) else first_batch
    
    if images.shape[0] == 0:
        return None

    batch_size = images.shape[0]
    indices = random.sample(range(batch_size), min(n_images, batch_size))
    selected_images = images[indices]
    return selected_images.to(device)

def get_accelerator() -> str:
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except AttributeError:
        pass
    return "cpu"

def save_reconstructions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    result_dir: str,
    output_filename: str = "reconstructions.png"
):
    result_path = Path(result_dir) / output_filename
    result_path.parent.mkdir(parents = True, exist_ok = True)

    selected_images = get_random_images(dataloader, 5, device)
    if selected_images is None or selected_images.numel() == 0:
        print("Could not generate reconstructions: no valid images found.")
        return

    model.eval().to(device)
    with torch.no_grad():
        mean, _ = model.encoder(selected_images)
        reconstructed_logits = model.decoder(mean)
        reconstructed_images = torch.sigmoid(reconstructed_logits)

    fig, axes = plt.subplots(len(selected_images), 2, figsize = (4, len(selected_images) * 2))
    
    for i in range(len(selected_images)):
        original_img = selected_images[i].cpu()
        reconstructed_img = reconstructed_images[i].cpu()

        if original_img.shape[0] == 1:
            original_np = original_img.squeeze().numpy()
            reconstructed_np = reconstructed_img.squeeze().numpy()
            cmap = 'gray'
        else:
            original_np = original_img.permute(1, 2, 0).numpy()
            reconstructed_np = reconstructed_img.permute(1, 2, 0).numpy()
            cmap = None

        axes[i, 0].imshow(original_np, cmap = cmap)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(reconstructed_np, cmap = cmap)
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(result_path)
    plt.close(fig)
    print(f"Reconstructions saved to {result_path}")

def compute_mig(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str, n_samples: int = 10000) -> float:
    model.eval().to(device)
    latents, factors = [], []

    for images, factor_vals in dataloader:
        images = images.to(device)
        with torch.no_grad():
            mean, _ = model.encoder(images)
        latents.append(mean.cpu().numpy())
        factors.append(factor_vals.cpu().numpy())
        if len(latents) * images.shape[0] >= n_samples:
            break

    latents = np.concatenate(latents, axis = 0)[:n_samples]
    factors = np.concatenate(factors, axis = 0)[:n_samples]

    n_latents, n_factors = latents.shape[1], factors.shape[1]
    mi_matrix = np.zeros((n_latents, n_factors))
    entropies = np.zeros(n_factors)

    for j in range(n_factors):
        unique_vals_count = len(np.unique(factors[:, j]))
        factor_bins = unique_vals_count if unique_vals_count <= 50 else 20
        factor_vals_binned = (
            factors[:, j].astype(int) if unique_vals_count <= 50 else 
            np.digitize(factors[:, j], np.linspace(factors[:, j].min(), factors[:, j].max(), factor_bins))
        )
        
        counts = np.bincount(factor_vals_binned) / len(factor_vals_binned)
        entropies[j] = entropy(counts, base = 2)

        for i in range(n_latents):
            latent_bins = 20
            latent_vals_binned = np.digitize(latents[:, i], np.linspace(latents[:, i].min(), latents[:, i].max(), latent_bins))
            mi_matrix[i, j] = mutual_info_score(latent_vals_binned, factor_vals_binned)

    sorted_mi = np.sort(mi_matrix, axis = 0)[::-1]
    mig_scores = (sorted_mi[0] - sorted_mi[1]) / np.maximum(entropies, 1e-10) 
    return np.mean(mig_scores[np.isfinite(mig_scores)])

class MIG(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if hasattr(trainer.datamodule, 'val_dataloader'):
            val_dataloader = trainer.datamodule.val_dataloader()
            device = pl_module.device
            mig_score = compute_mig(pl_module, val_dataloader, device = device)
            pl_module.log('val_mig', mig_score, prog_bar = True)
        else:
            print("Warning: datamodule does not have a 'val_dataloader' attribute. Cannot compute MIG.")