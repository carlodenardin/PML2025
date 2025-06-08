
import argparse
import torch
from pathlib import Path
import pytorch_lightning as pl
from src.dataset import DSpritesDataModule
from src.models.beta_vae import BetaVAE
from src.models.factor_vae import FactorVAE
from src.utils import save_reconstructions, compute_mig, get_accelerator, get_seed
from config import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type = str, required = True, choices = ['beta_vae', 'factor_vae'])
    parser.add_argument('--checkpoint', type = str)
    parser.add_argument('--num_workers', type = int, default = NUM_WORKERS)
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE)
    parser.add_argument('--train_val_test_split', type = list, default = TRAIN_VAL_TEST_SPLIT)
    
    return parser.parse_args()

def main():
    args = parse_args()
    result_dir, seed = get_seed(args.checkpoint)
    pl.seed_everything(seed)
    accellerator = get_accelerator()
    device = torch.device(accellerator if accellerator != "cpu" else "cpu")
    print(f"DEVICE: {device}")

    
    dm = DSpritesDataModule(
        data_dir = DIR_DSPRITES,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        train_val_test_split = args.train_val_test_split
    )
    dm.setup(stage = 'test')

    if args.model_type == 'beta_vae':
        model = BetaVAE.load_from_checkpoint(args.checkpoint)
        results_dir = Path(RESULTS_DIR_DSPRITES) / args.model_type / result_dir
        
    if args.model_type == 'factor_vae':
        model = FactorVAE.load_from_checkpoint(args.checkpoint)
        results_dir = Path(RESULTS_DIR_DSPRITES) / args.model_type / result_dir
    
    model.eval()
    model.to(device)
    results_dir.mkdir(parents = True, exist_ok = True)

    mig_score = compute_mig(model, dm.test_dataloader(), device = device)
    print(f"MIG Score: {mig_score}")

    # Save Visualizations
    save_reconstructions(model, dm.test_dataloader(), device, results_dir, output_filename = f"reconstructions_{mig_score:.2f}.png")

if __name__ == '__main__':
    main()
