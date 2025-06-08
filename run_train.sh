#!/bin/bash

# Test reconstruction error / MIG error for Beta VAE with different beta values
: '
BETA_VALUES="1 2 4 8 16 32 64"

for beta_value in $BETA_VALUES
do
  echo "Running Beta VAE (Beta = $beta_value)..."
  python train.py --model_type beta_vae --seed 1234 --beta $beta_value
done
'

# Test reconstruction error / MIG error for Factor VAE with different gamma values

GAMMA_VALUES="1 2 4 8 16 32 64"

for gamma_value in $GAMMA_VALUES
do
  echo "Running Factor VAE (Gamma = $gamma_value)..."
  python train.py --model_type factor_vae --seed 1234 --gamma $beta_value
done