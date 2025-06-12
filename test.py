import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch

from config import *
from src.dsprites import DSpritesDataModule
from src.models.beta_vae import BetaVAE
from src.models.factor_vae import FactorVAE
from src.mpi3d import MPI3DDataModule
from src.utils import *

# Parsing
def parse_args():
    """
        Parses command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type = str, required = True, choices = ['beta_vae', 'factor_vae'])
    parser.add_argument('--checkpoint', type = str, required = True)
    parser.add_argument('--num_workers', type = int, default = NUM_WORKERS)
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE)
    parser.add_argument('--train_val_test_split', type = list, default = TRAIN_VAL_TEST_SPLIT)
    parser.add_argument('--dataset', type = str, default = 'dsprites', choices = ['dsprites', 'mpi3d'])
    
    return parser.parse_args()

# Execution
def main():
    """
        Main function to load a model, evaluate it, and save results
    """
    # Setup and Initialization
    args = parse_args()
    result_dir, seed = get_seed(args.checkpoint)
    pl.seed_everything(seed)
    
    accelerator = get_accelerator()
    device = torch.device("cuda" if accelerator != "cpu" else "cpu")
    print(f"DEVICE: {device}")

    # Data Loading
    if args.dataset == 'dsprites':
        dm = DSpritesDataModule(
            data_dir = DIR_DSPRITES,
            batch_size = args.batch_size,
            num_workers = args.num_workers,
            train_val_test_split = args.train_val_test_split
        )
    elif args.dataset == 'mpi3d':
        dm = MPI3DDataModule(
            data_dir = DIR_MPI3D,
            batch_size = args.batch_size,
            num_workers = args.num_workers,
            train_val_test_split = args.train_val_test_split
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    dm.setup(stage = 'test')

    # Model loading
    if args.model_type == 'beta_vae':
        model = BetaVAE.load_from_checkpoint(args.checkpoint)
    elif args.model_type == 'factor_vae':
        model = FactorVAE.load_from_checkpoint(args.checkpoint)

    base_results_path = RESULTS_DIR_DSPRITES if args.dataset == 'dsprites' else RESULTS_DIR_MPI3D
    results_dir = Path(base_results_path) / args.model_type / result_dir
    
    model.eval()
    model.to(device)
    results_dir.mkdir(parents = True, exist_ok = True)

    # Evaluation and visualization
    mig_score = compute_mig(model, dm.test_dataloader(), device = device)
    print(f"MIG Score: {mig_score}")

    save_reconstructions(
        model, 
        dm.test_dataloader(), 
        device, 
        results_dir, 
        output_filename = f"reconstructions_{mig_score:.2f}.png"
    )

if __name__ == '__main__':
    main()