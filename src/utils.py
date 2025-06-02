import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_latent_traversals(model, dataloader, n_samples=5, n_traversals=8, latent_dim_to_traverse=None, traverse_range=(-2, 2), device='cpu'):
    """
    Visualizes latent space traversals for a trained VAE-based model.
    """
    model.eval()
    model.to(device)

    # Get a batch of data
    data_iter = iter(dataloader)
    samples = next(data_iter)
    if isinstance(samples, list): # If dataloader returns [data, target]
        samples = samples[0]
    samples = samples[:n_samples].to(device)

    # Encode samples to get latent representations
    mean, logvar = model.encoder(samples)
    
    if latent_dim_to_traverse is None:
        latent_dims_to_show = model.latent_dim
    else:
        latent_dims_to_show = latent_dim_to_traverse
        if not isinstance(latent_dims_to_show, list):
            latent_dims_to_show = [latent_dims_to_show]

    # Create traversal plot
    fig, axes = plt.subplots(n_samples * len(latent_dims_to_show), n_traversals + 1, figsize=(n_traversals +1, n_samples * len(latent_dims_to_show) * 1.2 ))
    if n_samples * len(latent_dims_to_show) == 1: # if single row of plots
        axes = np.expand_dims(axes, axis=0)


    with torch.no_grad():
        for i in range(n_samples):
            original_img_logits = model.decoder(mean[i:i+1])
            original_img = torch.sigmoid(original_img_logits).cpu().squeeze().numpy()
            
            current_z = mean[i].clone() # Use mean for traversal base

            for dim_idx, latent_dim_actual in enumerate(latent_dims_to_show):
                row_offset = (i * len(latent_dims_to_show)) + dim_idx
                
                # Plot original reconstruction
                ax = axes[row_offset, 0]
                ax.imshow(original_img, cmap='gray')
                ax.set_title(f"Orig (S{i+1})")
                ax.axis('off')

                # Traverse current latent dimension
                for k, val in enumerate(np.linspace(traverse_range[0], traverse_range[1], n_traversals)):
                    traversed_z = current_z.clone()
                    traversed_z[latent_dim_actual] = val 
                    
                    recon_logits = model.decoder(traversed_z.unsqueeze(0))
                    recon_img = torch.sigmoid(recon_logits).cpu().squeeze().numpy()
                    
                    ax = axes[row_offset, k + 1]
                    ax.imshow(recon_img, cmap='gray')
                    ax.set_title(f"z{latent_dim_actual}={val:.1f}")
                    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def get_accelerator():
    if torch.cuda.is_available():
        return "cuda"
    try:
        # Check for MPS (Apple Silicon)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except AttributeError: # Older PyTorch versions might not have mps
        pass
    return "cpu"