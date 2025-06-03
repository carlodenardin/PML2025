import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os
import time # Per timestamp univoci nelle directory di output

# Importa i tuoi moduli
from dataset import DSpritesDataModule
from models.vae import VAE
from models.factor_vae import FactorVAE
from utils import (
    visualize_reconstructions,
    save_individual_latent_traversal_grids,
    get_accelerator,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Esegue esperimenti VAE o FactorVAE.")
    
    # Argomenti generali dell'esperimento
    parser.add_argument('--model_type', type=str, required=True, choices=['vae', 'factor_vae'],
                        help="Tipo di modello da addestrare ('vae' o 'factor_vae').")
    parser.add_argument('--latent_dim', type=int, default=10, help="Dimensione dello spazio latente.")
    parser.add_argument('--epochs', type=int, default=50, help="Numero di epoche di addestramento.") # Aumentato default
    parser.add_argument('--batch_size', type=int, default=512, help="Dimensione del batch.")
    parser.add_argument('--num_workers', type=int, default=12, help="Numero di workers per il DataLoader.")
    parser.add_argument('--seed', type=int, default=1234, help="Seed per la riproducibilità.")
    parser.add_argument('--patience_early_stopping', type=int, default=10, help="Pazienza per l'early stopping.")
    parser.add_argument('--base_output_dir', type=str, default="experiment_outputs_cli",
                        help="Directory base per salvare tutti gli output dell'esperimento.")

    # Argomenti specifici per VAE
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate per VAE.")
    parser.add_argument('--beta', type=float, default=4.0, help="Peso beta per il termine KL del VAE.")

    # Argomenti specifici per FactorVAE
    parser.add_argument('--lr_vae', type=float, default=1e-4, help="Learning rate per la parte VAE di FactorVAE.")
    parser.add_argument('--lr_disc', type=float, default=1e-4, help="Learning rate per il Discriminatore di FactorVAE.")
    parser.add_argument('--gamma', type=float, default=35.0, help="Peso gamma per il termine TC di FactorVAE.")
    parser.add_argument('--disc_hidden_units', type=int, default=1000, help="Unità nascoste per strato del Discriminatore.")
    parser.add_argument('--disc_layers', type=int, default=6, help="Numero di strati del Discriminatore.")

    # Argomenti per la visualizzazione (eseguita dopo l'addestramento)
    parser.add_argument('--run_visualizations', action='store_true', help="Esegui visualizzazioni dopo l'addestramento.")
    parser.add_argument('--n_reconstruction_images', type=int, default=8, help="Numero di immagini per la visualizzazione delle ricostruzioni.")
    parser.add_argument('--n_images_for_static_traversals', type=int, default=3, help="Numero di immagini base per le griglie di traversata statiche.")
    parser.add_argument('--n_images_for_gif_traversals', type=int, default=1, help="Numero di immagini base per generare set di GIF di traversata (1 set per immagine).")
    parser.add_argument('--n_traversal_steps_per_dim', type=int, default=11, help="Numero di step per ogni traversata di dimensione latente.")
    parser.add_argument('--traversal_range_min', type=float, default=-2.5, help="Valore minimo del range di traversata.")
    parser.add_argument('--traversal_range_max', type=float, default=2.5, help="Valore massimo del range di traversata.")
    parser.add_argument('--gif_duration_per_frame', type=int, default=150, help="Durata (ms) per frame nelle GIF di traversata.")

    return parser.parse_args()

def main(args):
    pl.seed_everything(args.seed)
    current_accelerator = get_accelerator()
    device = torch.device(current_accelerator if current_accelerator != "cpu" else "cpu")
    print(f"Utilizzo del device: {device} (Accelerator: {current_accelerator})")

    # Crea una directory di output specifica per questa esecuzione
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_specific_output_dir = os.path.join(args.base_output_dir, args.model_type, f"run_seed{args.seed}_{timestamp}")
    if not os.path.exists(run_specific_output_dir):
        os.makedirs(run_specific_output_dir, exist_ok=True)
    print(f"Output dell'esperimento salvati in: {run_specific_output_dir}")

    # DataModule
    dm = DSpritesDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir="data/dsprites" # Puoi renderlo un argomento se necessario
    )

    # Modello
    model = None
    monitor_metric = None
    if args.model_type == 'vae':
        model = VAE(latent_dim=args.latent_dim, lr=args.lr, beta=args.beta)
        monitor_metric = 'val_loss'
    elif args.model_type == 'factor_vae':
        model = FactorVAE(
            latent_dim=args.latent_dim, lr_vae=args.lr_vae, lr_disc=args.lr_disc,
            gamma=args.gamma, disc_hidden_units=args.disc_hidden_units,
            disc_layers=args.disc_layers
        )
        monitor_metric = 'val_vae_loss' # Monitora la loss del VAE, non del discriminatore
    
    print(f"Modello {args.model_type} inizializzato.")
    print(f"Hyperparameters: {model.hparams if hasattr(model, 'hparams') else 'N/A'}")


    # Callbacks e Logger
    checkpoint_dir = os.path.join("checkpoints", args.model_type, f"run_seed{args.seed}_{timestamp}")
    log_dir = "logs" # TensorBoardLogger creerà logs/model_type/run_seed...

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{{epoch}}-{{{monitor_metric}:.2f}}",
        save_top_k=1,
        verbose=True,
        monitor=monitor_metric,
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=args.patience_early_stopping,
        verbose=True,
        mode='min'
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=args.model_type,
        version=f"seed{args.seed}_{timestamp}" # Crea una sottocartella unica per questa esecuzione
    )
    print(f"Checkpoints salvati in: {checkpoint_dir}")
    print(f"Log di TensorBoard in: {os.path.join(log_dir, args.model_type, f'seed{args.seed}_{timestamp}')}")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=current_accelerator,
        devices=1 if current_accelerator != "cpu" else None,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=tensorboard_logger,
        enable_progress_bar=True # O False se vuoi meno output verboso per script automatici
    )

    # Addestramento
    print(f"Inizio addestramento per {args.model_type} per {args.epochs} epoche...")
    trainer.fit(model, datamodule=dm)
    print("Addestramento completato.")

    # Carica il miglior modello per la visualizzazione
    trained_model_for_viz = None
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        print(f"Caricamento del miglior modello da: {best_model_path}")
        if args.model_type == 'vae':
            trained_model_for_viz = VAE.load_from_checkpoint(best_model_path)
        else: # factor_vae
            trained_model_for_viz = FactorVAE.load_from_checkpoint(best_model_path)
        
        if trained_model_for_viz:
            trained_model_for_viz.to(device)
            trained_model_for_viz.eval()
            print("Miglior modello caricato e impostato in modalità evaluazione.")
        else:
            print(f"Errore nel caricamento del modello da {best_model_path}")
    else:
        print(f"Nessun checkpoint del miglior modello trovato in '{best_model_path}'. Visualizzazioni saltate o useranno il modello in memoria (se disponibile).")
        # Fallback al modello in memoria se l'addestramento è appena terminato
        if trainer.model:
            print("Utilizzo del modello corrente in memoria (post-addestramento) per le visualizzazioni.")
            trained_model_for_viz = trainer.model 
            trained_model_for_viz.to(device)
            trained_model_for_viz.eval()

    # Visualizzazioni (se richieste e se il modello è disponibile)
    if args.run_visualizations and trained_model_for_viz:
        print("\nEsecuzione visualizzazioni...")
        dm.setup(stage='test') # Assicurati che il test dataloader sia pronto
        test_dataloader = dm.test_dataloader()

        if test_dataloader and len(test_dataloader.dataset) > 0:
            current_epoch_str = str(trained_model_for_viz.current_epoch if hasattr(trained_model_for_viz, 'current_epoch') else 'final')

            # 1. Visualizzazione Ricostruzioni
            print("\nVisualizzazione e salvataggio ricostruzioni...")
            recon_output_dir = os.path.join(run_specific_output_dir, "reconstructions")
            visualize_reconstructions(
                trained_model_for_viz, test_dataloader,
                n_images=args.n_reconstruction_images, device=device,
                output_dir=recon_output_dir,
                output_filename=f"reconstructions_ep{current_epoch_str}.png"
            )

            # 2. Visualizzazione Griglie di Traversata Statiche
            print("\nGenerazione e salvataggio griglie di traversata statiche...")
            static_traversal_dir = os.path.join(run_specific_output_dir, "static_traversals")
            save_individual_latent_traversal_grids(
                trained_model_for_viz, test_dataloader,
                n_images_to_show=args.n_images_for_static_traversals,
                n_traversal_steps=args.n_traversal_steps_per_dim,
                traverse_range=(args.traversal_range_min, args.traversal_range_max),
                device=device, output_dir=static_traversal_dir,
                filename_prefix=f"static_traversal_ep{current_epoch_str}_img_"
            )
            
        else:
            print("Test dataloader non disponibile o vuoto. Visualizzazioni saltate.")
    elif args.run_visualizations:
        print("Nessun modello addestrato disponibile per la visualizzazione.")

    print(f"\nEsperimento completato per {args.model_type}. Controlla la directory '{run_specific_output_dir}'.")

if __name__ == '__main__':
    args = parse_args()
    main(args)