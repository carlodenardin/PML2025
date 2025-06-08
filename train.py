import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from config import *
from src.dataset import DSpritesDataModule
from src.models.beta_vae import BetaVAE
from src.models.factor_vae import FactorVAE
from src.utils import get_accelerator, MIG

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type = str, required = True, choices = ['beta_vae', 'factor_vae'])
    parser.add_argument('--beta', type = float, default = BETA)
    parser.add_argument('--gamma', type = float, default = GAMMA)
    parser.add_argument('--seed', type = float, default = SEED)
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
    
    return parser.parse_args()

def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    accellerator = get_accelerator()
    device = torch.device(accellerator if accellerator != "cpu" else "cpu")
    print(f"DEVICE: {device}")

    dm = DSpritesDataModule(
        data_dir = DIR_DSPRITES,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        train_val_test_split = args.train_val_test_split
    )

    if args.model_type == 'beta_vae':
        model = BetaVAE(latent_dim = args.latent_dim, lr = args.lr_vae, beta = args.beta)
        checkpoints_dir = Path(CHECKPOINTS_DIR_DISPRITES) / args.model_type / f"seed_{args.seed}_beta_{args.beta}"
        tensorboard_logger = TensorBoardLogger(save_dir = LOGS_DIR_DISPRITES, name = args.model_type, version = f"seed_{args.seed}_beta_{args.beta}")

    if args.model_type == 'factor_vae':
        model = FactorVAE(
            latent_dim = args.latent_dim,
            lr_vae = args.lr_vae,
            lr_disc = args.lr_disc,
            gamma = args.gamma,
            disc_hidden_units = args.hidden_units_d,
            disc_layers = args.num_layers_d
        )
        checkpoints_dir = Path(CHECKPOINTS_DIR_DISPRITES) / args.model_type / f"seed_{args.seed}_gamma_{args.gamma}"
        tensorboard_logger = TensorBoardLogger(save_dir = LOGS_DIR_DISPRITES, name = args.model_type, version = f"seed_{args.seed}_gamma_{args.gamma}")
    
    # Callbacks
    checkpoints = ModelCheckpoint(
        dirpath = checkpoints_dir,
        filename = f"{{epoch}}-{{{MONITOR_METRIC}:.2f}}",
        save_top_k = 1,
        verbose = True,
        monitor = MONITOR_METRIC,
        mode = "min"
    )

    stopping = EarlyStopping(
        monitor = MONITOR_METRIC,
        patience = args.patience,
        verbose = True,
        mode = "min"
    )

    mig = MIG()
    
    trainer = pl.Trainer(
        max_epochs = args.epochs,
        accelerator = accellerator,
        devices=1 if accellerator != "cpu" else None,
        callbacks=[
            checkpoints,
            stopping,
            mig
        ],
        logger = tensorboard_logger,
        enable_progress_bar = True,
        precision = args.precision,
    )

    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    main()
