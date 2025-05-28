# Anomaly Detection

A Probabilistic Machine Learning project comparing Autoencoder, VAE, and Diffusion Models for anomaly detection.

## Setup

1. Install Miniconda.
2. Run `./setup.sh` to create the environment.
3. Download the dataset [DRAFT]

## Usage

- Train: `python src/train.py --model ae --epochs 10 --data_dir data/chest_xray`
- Evaluate: `python src/evaluate.py --model ae --data_dir data/chest_xray`

## Structure

- `data/`: Dataset.
- `src/`: Python code (dataset, models, training).
- `experiments/`: Outputs and checkpoints.
