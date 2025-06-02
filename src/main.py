import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataset import DSpritesDataModule
from models.vae import VAE
from models.factor_vae import FactorVAE
from utils import visualize_latent_traversals, get_accelerator

def main(args):
    pl.seed_everything(args.seed)
    device = torch.device(get_accelerator())
    print(f"Using device: {device}")

    dm = DSpritesDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    dm.prepare_data()
    dm.setup(stage='fit')


    if args.model_type == 'vae':
        model = VAE(latent_dim=args.latent_dim, lr=args.lr, beta=args.beta)
    elif args.model_type == 'factor_vae':
        model = FactorVAE(latent_dim=args.latent_dim, 
                          lr_vae=args.lr_vae, 
                          lr_disc=args.lr_disc, 
                          gamma=args.gamma,
                          disc_hidden_units=args.disc_hidden_units,
                          disc_layers=args.disc_layers)
    else:
        raise ValueError("Invalid model_type specified.")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{args.model_type}",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor='val_loss', # or 'val_vae_loss' for FactorVAE
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss', # or 'val_vae_loss' for FactorVAE
        patience=args.patience,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=get_accelerator(),
        devices=1 if get_accelerator() != "cpu" else None,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=pl.loggers.TensorBoardLogger(f"logs/{args.model_type}/"),
    )

    print(f"Starting training for {args.model_type}...")
    trainer.fit(model, dm)
    print("Training finished.")

    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best model from: {best_model_path}")
    if args.model_type == 'vae':
        trained_model = VAE.load_from_checkpoint(best_model_path)
    else:
        trained_model = FactorVAE.load_from_checkpoint(best_model_path)
    
    trained_model.to(device)
    trained_model.eval()

    print("Visualizing latent traversals...")
    val_dataloader = dm.val_dataloader()
    
    dims_to_show = list(range(min(args.latent_dim, 5)))
    
    visualize_latent_traversals(
        trained_model, 
        val_dataloader, 
        n_samples=3, 
        n_traversals=7, 
        latent_dim_to_traverse=dims_to_show,
        traverse_range=(-3, 3),
        device=device
    )
    print("Visualization complete. Check the displayed plot.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train VAE or FactorVAE on dSprites.")
    parser.add_argument('--model_type', type=str, required=True, choices=['vae', 'factor_vae'], help="Type of model to train.")
    parser.add_argument('--latent_dim', type=int, default=10, help="Dimensionality of the latent space.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping.")

    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for VAE.")
    parser.add_argument('--beta', type=float, default=1.0, help="Beta hyperparameter for VAE KL divergence term.")

    parser.add_argument('--lr_vae', type=float, default=1e-4, help="Learning rate for FactorVAE's VAE components.")
    parser.add_argument('--lr_disc', type=float, default=1e-4, help="Learning rate for FactorVAE's Discriminator.")
    parser.add_argument('--gamma', type=float, default=35.0, help="Gamma hyperparameter for FactorVAE TC term (e.g., values from paper Fig 6 [cite: 146]).")
    parser.add_argument('--disc_hidden_units', type=int, default=1000, help="Hidden units per layer in Discriminator.")
    parser.add_argument('--disc_layers', type=int, default=6, help="Number of hidden layers in Discriminator.")
    
    args = parser.parse_args()
    main(args)