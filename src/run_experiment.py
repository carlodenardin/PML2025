import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import time
from config import ExperimentConfig
from dataset import DSpritesDataModule
from models.vae import VAE
from models.factor_vae import FactorVAE
from utils import get_accelerator, run_visualizations, compute_mig

def parse_args():
    parser = argparse.ArgumentParser(description="Esegue esperimenti VAE o FactorVAE.")
    parser.add_argument('--model_type', type=str, required=True, choices=['vae', 'factor_vae'],
                        help="Tipo di modello da addestrare ('vae' o 'factor_vae').")
    return parser.parse_args()

def main():
    args = parse_args()
    config = ExperimentConfig()
    pl.seed_everything(config.seed)
    current_accelerator = get_accelerator()
    device = torch.device(current_accelerator if current_accelerator != "cpu" else "cpu")
    print(f"Utilizzo del device: {device} (Accelerator: {current_accelerator})")

    # Configura directory di output
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(config.base_output_dir) / args.model_type / f"run_seed{config.seed}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dell'esperimento salvati in: {output_dir}")

    # DataModule
    dm = DSpritesDataModule(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_val_test_split=config.train_val_test_split
    )

    # Modello
    model = (
        VAE(latent_dim=config.latent_dim, lr=config.lr_vae, beta=config.beta)
        if args.model_type == 'vae'
        else FactorVAE(
            latent_dim=config.latent_dim, lr_vae=config.lr_vae, lr_disc=config.lr_disc,
            gamma=config.gamma, disc_hidden_units=config.disc_hidden_units,
            disc_layers=config.disc_layers
        )
    )
    monitor_metric = 'val_loss' if args.model_type == 'vae' else 'val_vae_loss'
    print(f"Modello {args.model_type} inizializzato.")
    print(f"Hyperparameters: {model.hparams}")

    # Callbacks e Logger
    checkpoint_dir = Path(config.checkpoint_dir) / args.model_type / f"run_seed{config.seed}_{timestamp}"
    tensorboard_logger = TensorBoardLogger(save_dir=config.log_dir, name=args.model_type, version=f"seed{config.seed}_{timestamp}")
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator=current_accelerator,
        devices=1 if current_accelerator != "cpu" else None,
        callbacks=[
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename=f"{{epoch}}-{{{monitor_metric}:.2f}}",
                save_top_k=1,
                verbose=True,
                monitor=monitor_metric,
                mode='min'
            ),
            EarlyStopping(
                monitor=monitor_metric,
                patience=config.patience_early_stopping,
                verbose=True,
                mode='min'
            )
        ],
        logger=tensorboard_logger,
        enable_progress_bar=True,
        precision=config.precision
    )

    # Addestramento
    print(f"Inizio addestramento per {args.model_type} per {config.epochs} epoche...")
    trainer.fit(model, datamodule=dm)
    print("Addestramento completato.")

    # Carica il miglior modello
    best_model = model
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path and Path(best_model_path).exists():
        print(f"Caricamento del miglior modello da: {best_model_path}")
        best_model = VAE.load_from_checkpoint(best_model_path) if args.model_type == 'vae' else FactorVAE.load_from_checkpoint(best_model_path)
        best_model.to(device).eval()
    else:
        print("Utilizzo del modello corrente in memoria per le visualizzazioni.")
        best_model.to(device).eval()

    # Visualizzazioni e metriche
    if config.run_visualizations:
        print("\nEsecuzione visualizzazioni e calcolo metriche...")
        dm.setup(stage='test')
        run_visualizations(best_model, dm.test_dataloader(), config, output_dir, device)
        mig_score = compute_mig(best_model, dm.test_dataloader(), device=device)
        print(f"MIG Score: {mig_score:.4f}")

    print(f"\nEsperimento completato per {args.model_type}. Controlla la directory '{output_dir}'.")

if __name__ == '__main__':
    main()
