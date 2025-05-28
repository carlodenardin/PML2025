# utils.py
import os
import torch
from torchvision.utils import save_image, make_grid

def save_generated_images(images, epoch, results_dir, filename_prefix='generated'):
    """
    Salva un grid di immagini generate.

    :param images: Tensore delle immagini generate (formato Batch x C x H x W).
    :param epoch: Numero dell'epoca corrente.
    :param results_dir: Directory dove salvare le immagini.
    :param filename_prefix: Prefisso per il nome del file.
    """
    grid = make_grid(images, nrow=8, padding=2, normalize=True)
    save_image(grid, os.path.join(results_dir, f'{filename_prefix}_epoch_{epoch}.png'))
    print(f"Immagini generate salvate in {results_dir}")

def save_reconstruction_comparison(original_images, reconstructed_images, epoch, results_dir):
    """
    Salva un grid di immagini originali e le loro ricostruzioni per confronto.

    :param original_images: Tensore delle immagini originali.
    :param reconstructed_images: Tensore delle immagini ricostruite.
    :param epoch: Numero dell'epoca corrente.
    :param results_dir: Directory dove salvare le immagini.
    """
    comparison = torch.cat([original_images, reconstructed_images])
    grid_comparison = make_grid(comparison, nrow=original_images.shape[0], padding=2, normalize=True)
    save_image(grid_comparison, os.path.join(results_dir, f'reconstructions_epoch_{epoch}.png'))
    print(f"Confronto ricostruzioni salvato in {results_dir}")

def setup_results_dir(results_dir):
    """
    Crea la directory per i risultati se non esiste.
    """
    os.makedirs(results_dir, exist_ok=True)
    print(f"Directory per i risultati: {results_dir}")