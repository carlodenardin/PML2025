import argparse
import torch
from pathlib import Path
import pytorch_lightning as pl
from config import ExperimentConfig
from dataset import DSpritesDataModule
from models.vae import VAE
from models.factor_vae import FactorVAE
from utils import run_visualizations, compute_mig

def main():
    # Parser degli argomenti
    parser = argparse.ArgumentParser(description="Esegue test su un modello addestrato (VAE o FactorVAE).")
    parser.add_argument('--model_type', type=str, choices=['vae', 'factor_vae'], required=True,
                        help="Tipo di modello: 'vae' o 'factor_vae'")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Percorso al file di checkpoint del modello")
    parser.add_argument('--seed', type=int, default=11, help="Seed per la riproducibilità")
    args = parser.parse_args()

    # Imposta il seed
    pl.seed_everything(args.seed, workers=True)

    # Carica la configurazione
    config = ExperimentConfig()

    # Determina il dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilizzo dispositivo: {device}")

    # Inizializza il DataModule
    dm = DSpritesDataModule(data_dir=config.data_dir, batch_size=config.batch_size,
                            num_workers=config.num_workers)
    dm.setup(stage='test')

    # Carica il modello dal checkpoint
    if args.model_type == 'vae':
        model = VAE.load_from_checkpoint(args.checkpoint_path, config=config)
    elif args.model_type == 'factor_vae':
        model = FactorVAE.load_from_checkpoint(args.checkpoint_path, config=config)
    else:
        raise ValueError(f"Model type {args.model_type} non supportato.")

    model.eval()
    model.to(device)

    # Directory di output
    output_dir = Path(config.base_output_dir) / args.model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # Esegue visualizzazioni
    print("Esecuzione delle visualizzazioni...")
    run_visualizations(model, dm.test_dataloader(), config, output_dir, device)

    # Calcola il MIG score
    print("Calcolo del MIG score...")
    mig_score = compute_mig(model, dm.test_dataloader(), device=device)
    print(f"MIG Score: {mig_score:.4f}")

if __name__ == '__main__':
    main()