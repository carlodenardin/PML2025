import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import random
import imageio
from sklearn.metrics import mutual_info_score

def get_random_images(dataloader, n_images, device):
    """Raccoglie n_images casuali dal dataloader e le trasferisce al device."""
    all_images = []
    num_batches = (n_images + dataloader.batch_size - 1) // dataloader.batch_size
    for i, batch in enumerate(dataloader):
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        all_images.append(images)
        if i + 1 >= num_batches:
            break
    if not all_images:
        print("Nessuna immagine caricata dal dataloader.")
        return None
    all_images = torch.cat(all_images, dim=0)
    n_available = len(all_images)
    indices = random.sample(range(n_available), min(n_images, n_available))
    return all_images[indices].to(device)

def visualize_reconstructions(model, dataloader, n_images=10, device='cpu',
                              output_dir="reconstruction_images",
                              output_filename="reconstructions.png"):
    """Visualizza e salva le ricostruzioni delle immagini."""
    model.eval()
    model.to(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_images = get_random_images(dataloader, n_images, device)
    if selected_images is None:
        return

    with torch.no_grad():
        mean, logvar = model.encoder(selected_images)
        z = model._reparameterize(mean, logvar) if hasattr(model, '_reparameterize') else mean
        reconstructed_logits = model.decoder(z)
        reconstructed_images = torch.sigmoid(reconstructed_logits)

    fig, axes = plt.subplots(len(selected_images), 2, figsize=(4, len(selected_images) * 2), squeeze=False)
    for i in range(len(selected_images)):
        axes[i, 0].imshow(selected_images[i].cpu().squeeze().numpy(), cmap='gray')
        axes[i, 0].set_title(f"Originale {i+1}")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(reconstructed_images[i].cpu().squeeze().numpy(), cmap='gray')
        axes[i, 1].set_title(f"Ricostruita {i+1}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    save_path = output_dir / output_filename
    try:
        plt.savefig(save_path)
        print(f"Immagine delle ricostruzioni salvata in: {save_path}")
    except Exception as e:
        print(f"Errore durante il salvataggio delle ricostruzioni: {e}")
    plt.close(fig)

def save_individual_latent_traversal_grids(model, source_images, n_images_to_show=4, n_traversal_steps=10, traverse_range=(-2.0, 2.0), device='cpu', output_dir="traversal_images", filename_prefix="traversal"):
    """Salva immagini di traversate latenti per ogni immagine sorgente e dimensione latente."""
    model.eval()
    model.to(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if source_images is None:
        print("Nessuna immagine sorgente fornita.")
        return

    source_images = source_images[:n_images_to_show].to(device)  # Usa immagini fornite direttamente
    latent_dim = model.latent_dim if hasattr(model, 'latent_dim') else source_images.shape[1]
    
    with torch.no_grad():
        mean, logvar = model.encoder(source_images)
        z = model._reparameterize(mean, logvar) if hasattr(model, '_reparameterize') else mean

    for img_idx in range(len(source_images)):
        for dim in range(latent_dim):
            fig, axes = plt.subplots(1, n_traversal_steps, figsize=(n_traversal_steps * 2, 2))
            axes = axes.flatten() if n_traversal_steps > 1 else [axes]
            traversal_values = torch.linspace(traverse_range[0], traverse_range[1], n_traversal_steps, device=device)
            
            for step, value in enumerate(traversal_values):
                z_traversal = z[img_idx:img_idx+1].clone()
                z_traversal[0, dim] = value
                with torch.no_grad():
                    recon_logits = model.decoder(z_traversal)
                    recon_image = torch.sigmoid(recon_logits)
                axes[step].imshow(recon_image[0].cpu().squeeze().numpy(), cmap='gray')
                axes[step].set_title(f'z[{dim}]={value:.2f}')
                axes[step].axis('off')
            
            plt.tight_layout()
            save_path = output_dir / f"{filename_prefix}{img_idx}_dim{dim}.png"
            try:
                plt.savefig(save_path)
                print(f"Traversata salvata in: {save_path}")
            except Exception as e:
                print(f"Errore durante il salvataggio della traversata: {e}")
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

def run_visualizations(model, dataloader, config, output_dir, device):
    """Esegue tutte le visualizzazioni (ricostruzioni e traversate latenti)."""
    output_dir = Path(output_dir)
    model.eval()
    model.to(device)

    # Ricostruzioni
    recon_dir = output_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)
    images = get_random_images(dataloader, config.n_reconstruction_images, device)
    if images is not None:
        visualize_reconstructions(
            model, images, n_images=len(images), device=device,
            output_dir=recon_dir, output_filename=f"reconstructions_ep{model.current_epoch or 'final'}.png"
        )

    # Traversate latenti
    traversal_dir = output_dir / "static_traversals"
    traversal_dir.mkdir(parents=True, exist_ok=True)
    images = get_random_images(dataloader, config.n_images_for_static_traversals, device)
    if images is not None:
        save_individual_latent_traversal_grids(
            model, images, n_images_to_show=len(images),
            n_traversal_steps=config.n_traversal_steps_per_dim,
            traverse_range=(config.traversal_range_min, config.traversal_range_max),
            device=device, output_dir=traversal_dir,
            filename_prefix=f"static_traversal_ep{model.current_epoch or 'final'}_img_"
        )
