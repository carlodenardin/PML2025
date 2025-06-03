import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import imageio

# ... (la funzione visualize_reconstructions e get_accelerator rimangono qui) ...
# La vecchia funzione visualize_latent_traversals può essere rimossa o commentata
# se questa nuova la sostituisce per il tuo scopo.

def save_individual_latent_traversal_grids(model, dataloader,
                                           n_images_to_show=3,
                                           n_traversal_steps=7,
                                           traverse_range=(-2.5, 2.5),
                                           device='cpu',
                                           output_dir="traversal_grids_per_image",
                                           filename_prefix="traversal_grid_img_"):
    """
    Per n_images_to_show immagini selezionate casualmente dal dataloader (test set):
    Crea e salva una griglia di immagini separata per ciascuna.
    Ogni griglia mostra le traversate per ogni dimensione latente.
    Righe: dimensioni latenti.
    Colonne: valori di traversata per quella dimensione latente.
    """
    model.eval()
    model.to(device)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Creata directory per le griglie di traversata: {output_dir}")

    # 1. Raccogli immagini dal dataloader
    all_images_from_loader = []
    # Assicurati di caricare abbastanza immagini se n_images_to_show è grande,
    # o almeno un batch se n_images_to_show è piccolo.
    num_batches_to_fetch = (n_images_to_show + dataloader.batch_size -1) // dataloader.batch_size
    
    for i, batch in enumerate(dataloader):
        if isinstance(batch, list):
            images_in_batch = batch[0]
        else:
            images_in_batch = batch
        all_images_from_loader.append(images_in_batch)
        if i + 1 >= num_batches_to_fetch:
            break
    
    if not all_images_from_loader:
        print("Nessuna immagine caricata dal dataloader per le traversate.")
        return

    all_images_from_loader = torch.cat(all_images_from_loader, dim=0)

    # 2. Seleziona n_images_to_show casualmente
    if len(all_images_from_loader) >= n_images_to_show:
        indices = random.sample(range(len(all_images_from_loader)), n_images_to_show)
        source_images = all_images_from_loader[indices].to(device)
    elif len(all_images_from_loader) > 0:
        source_images = all_images_from_loader[:n_images_to_show].to(device) # Prendi quelle disponibili
        print(f"Attenzione: richieste {n_images_to_show} immagini, ma solo {len(source_images)} disponibili nel batch caricato.")
        if len(source_images) == 0: return
    else:
        print("Non abbastanza immagini nel dataloader.")
        return

    latent_dim = model.latent_dim
    traversal_values = np.linspace(traverse_range[0], traverse_range[1], n_traversal_steps)

    # 3. Loop per ogni immagine sorgente selezionata
    for img_idx in range(source_images.shape[0]):
        original_image = source_images[img_idx:img_idx+1] # Mantieni la dimensione del batch

        with torch.no_grad():
            # Codifica l'immagine originale per ottenere z_mean come base
            z_mean_orig, _ = model.encoder(original_image)

            # Plot: righe = dimensioni latenti, colonne = step di traversata
            fig, axes = plt.subplots(latent_dim, n_traversal_steps,
                                     figsize=(n_traversal_steps * 1.5, latent_dim * 1.5),
                                     squeeze=False) # squeeze=False è importante
            
            fig.suptitle(f"Traversate Latenti per Immagine Test {img_idx+1}", fontsize=10)

            for i in range(latent_dim):  # Loop su ogni dimensione latente (righe)
                z_base_for_dim_i = z_mean_orig.clone() # Usa la media dell'immagine originale come base

                for j, val in enumerate(traversal_values):  # Loop sui valori di traversata (colonne)
                    z_traversed = z_base_for_dim_i.clone()
                    z_traversed[0, i] = val  # Modifica la i-esima dimensione latente

                    reconstructed_logits = model.decoder(z_traversed)
                    reconstructed_image = torch.sigmoid(reconstructed_logits)
                    
                    ax = axes[i, j]
                    ax.imshow(reconstructed_image.cpu().squeeze().numpy(), cmap='gray')
                    ax.axis('off')
                    if i == 0: # Titolo della colonna solo per la prima riga
                        ax.set_title(f"{val:.1f}", fontsize=8)
                
                axes[i, 0].text(-10, axes[i,0].get_images()[0].get_array().shape[0]//2, f"z_{i}", 
                                va='center', ha='right', fontsize=8, rotation=0)


            plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Aggiusta per suptitle

            # Salva la figura per l'immagine corrente
            current_filename = f"{filename_prefix}{img_idx+1}.png"
            save_path = os.path.join(output_dir, current_filename)
            try:
                plt.savefig(save_path)
                print(f"Griglia di traversata per immagine {img_idx+1} salvata in: {save_path}")
            except Exception as e:
                print(f"Errore durante il salvataggio della griglia di traversata per immagine {img_idx+1}: {e}")
            plt.close(fig)

# ----- La funzione visualize_reconstructions può rimanere com'è -----
def visualize_reconstructions(model, dataloader, n_images=10, device='cpu',
                              output_dir="reconstruction_images",
                              output_filename="reconstructions.png"):
    # ... (codice esistente per visualize_reconstructions) ...
    model.eval()
    model.to(device)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Creata directory: {output_dir}")

    all_images_from_loader = []
    num_batches_to_fetch = (n_images + dataloader.batch_size -1) // dataloader.batch_size
    
    for i, batch in enumerate(dataloader):
        if isinstance(batch, list):
            images_in_batch = batch[0]
        else:
            images_in_batch = batch
        all_images_from_loader.append(images_in_batch)
        if i + 1 >= num_batches_to_fetch:
            break
    
    if not all_images_from_loader:
        print("Nessuna immagine caricata dal dataloader per la visualizzazione delle ricostruzioni.")
        return

    all_images_from_loader = torch.cat(all_images_from_loader, dim=0)

    if len(all_images_from_loader) > n_images:
        indices = random.sample(range(len(all_images_from_loader)), n_images)
        selected_images = all_images_from_loader[indices].to(device)
    elif len(all_images_from_loader) > 0:
        selected_images = all_images_from_loader[:n_images].to(device)
        if len(selected_images) == 0: 
            print("Non abbastanza immagini nel dataloader per visualizzare le ricostruzioni.")
            return
    else:
        print("Non abbastanza immagini nel dataloader per visualizzare le ricostruzioni.")
        return
    
    n_actual_images = selected_images.shape[0]
    if n_actual_images == 0:
        print("Nessuna immagine selezionata per la visualizzazione.")
        return

    with torch.no_grad():
        mean, logvar = model.encoder(selected_images)
        if hasattr(model, '_reparameterize'):
            z = model._reparameterize(mean, logvar)
        else:
            z = mean
        reconstructed_logits = model.decoder(z)
        reconstructed_images = torch.sigmoid(reconstructed_logits)

    fig, axes = plt.subplots(n_actual_images, 2, figsize=(4, n_actual_images * 2), squeeze=False)
    # if n_actual_images == 1: 
    #      axes = np.expand_dims(axes, axis=0) # Già gestito da squeeze=False

    for i in range(n_actual_images):
        ax = axes[i, 0]
        ax.imshow(selected_images[i].cpu().squeeze().numpy(), cmap='gray')
        ax.set_title(f"Originale {i+1}")
        ax.axis('off')

        ax = axes[i, 1]
        ax.imshow(reconstructed_images[i].cpu().squeeze().numpy(), cmap='gray')
        ax.set_title(f"Ricostruita {i+1}")
        ax.axis('off')

    plt.tight_layout()

    save_path = os.path.join(output_dir, output_filename)
    try:
        plt.savefig(save_path)
        print(f"Immagine delle ricostruzioni salvata in: {save_path}")
    except Exception as e:
        print(f"Errore durante il salvataggio dell'immagine delle ricostruzioni: {e}")
    plt.close(fig)

def get_accelerator():
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except AttributeError:
        pass
    return "cpu"