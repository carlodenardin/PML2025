import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from config import *
from src.dsprites import DSpritesDataModule
from src.models.beta_vae import BetaVAE
from src.models.factor_vae import FactorVAE
from src.mpi3d import MPI3DDataModule
from src.utils import *

# Parsing
def parse_args():
    """
        Parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type = str, required = True, choices = ['beta_vae', 'factor_vae'])
    parser.add_argument('--beta', type = float, default = BETA)
    parser.add_argument('--gamma', type = float, default = GAMMA)
    parser.add_argument('--seed', type = int, default = SEED)
    parser.add_argument('--epochs', type = int, default = EPOCHS)
    parser.add_argument('--patience', type = int, default = PATIENCE)
    parser.add_argument('--precision', type = str, default = PRECISION)
    parser.add_argument('--lr_vae', type = float, default = LR_VAE)
    parser.add_argument('--lr_disc', type = float, default = LR_DISC)
    parser.add_argument('--latent_dim', type = int, default = LATENT_DIM)
    parser.add_argument('--hidden_units_d', type = int, default = HIDDEN_UNITS_D)
    parser.add_argument('--num_layers_d', type = int, default = NUM_LAYERS_D)
    parser.add_argument('--num_workers', type = int, default = NUM_WORKERS)
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE)
    parser.add_argument('--train_val_test_split', type = list, default = TRAIN_VAL_TEST_SPLIT)
    parser.add_argument('--dataset', type = str, default = 'dsprites', choices = ['dsprites', 'mpi3d'])
    
    return parser.parse_args()

# Execution
def main():
    """
        Main function to set up and run the training pipeline.
    """
    # Setup and Initialization
    args = parse_args()
    pl.seed_everything(args.seed)
    
    accelerator = get_accelerator()
    device = torch.device("cuda" if accelerator != "cpu" else "cpu")
    print(f"--- Training on device: {device} ---")

    # Data Loading
    if args.dataset == 'dsprites':
        dm = DSpritesDataModule(
            data_dir = DIR_DSPRITES,
            batch_size = args.batch_size,
            num_workers = args.num_workers,
            train_val_test_split = args.train_val_test_split
        )
        in_channels = 1
        reconstruction_loss = "bce"
    elif args.dataset == 'mpi3d':
        dm = MPI3DDataModule(
            data_dir = DIR_MPI3D,
            batch_size = args.batch_size,
            num_workers = args.num_workers,
            train_val_test_split = args.train_val_test_split
        )
        in_channels = 3
        reconstruction_loss = "mse"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Model Initialization
    if args.model_type == 'beta_vae':
        model = BetaVAE(
            latent_dim = args.latent_dim,
            lr = args.lr_vae,
            beta = args.beta,
            in_channels = in_channels,
            out_channels = in_channels,
            rec_loss = reconstruction_loss
        )
        version_name = f"seed_{args.seed}_beta_{args.beta}"
    elif args.model_type == 'factor_vae':
        model = FactorVAE(
            latent_dim = args.latent_dim,
            lr_vae = args.lr_vae,
            lr_disc = args.lr_disc,
            gamma = args.gamma,
            disc_hidden_units = args.hidden_units_d,
            disc_layers = args.num_layers_d,
            in_channels = in_channels,
            out_channels = in_channels,
            rec_loss = reconstruction_loss
        )
        version_name = f"seed_{args.seed}_gamma_{args.gamma}"

    # Callbacks and Logger Setup
    checkpoints_base_dir = CHECKPOINTS_DIR_DSPRITES if args.dataset == 'dsprites' else CHECKPOINTS_DIR_MPI3D
    logs_base_dir = LOGS_DIR_DSPRITES if args.dataset == 'dsprites' else LOGS_DIR_MPI3D

    checkpoints_dir = Path(checkpoints_base_dir) / args.model_type / version_name
    tensorboard_logger = TensorBoardLogger(save_dir = logs_base_dir, name = args.model_type, version = version_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath = checkpoints_dir,
        filename = f"{{epoch}}-{{{MONITOR_METRIC}:.2f}}",
        save_top_k = 1,
        verbose = True,
        monitor = MONITOR_METRIC,
        mode = "min"
    )
    
    early_stopping_callback = EarlyStopping(
        monitor = MONITOR_METRIC,
        patience = args.patience,
        verbose = True,
        mode = "min"
    )

    mig_callback = MIG()

    # Trainer Setup
    trainer = pl.Trainer(
        max_epochs = args.epochs,
        accelerator = accelerator,
        devices = 1 if accelerator != "cpu" else None,
        callbacks = [
            checkpoint_callback,
            early_stopping_callback,
            mig_callback
        ],
        logger = tensorboard_logger,
        enable_progress_bar = True,
        precision = args.precision,
    )

    # Training
    trainer.fit(model, datamodule = dm)


if __name__ == '__main__':
    main()