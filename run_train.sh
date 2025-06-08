#!/bin/bash

# Test reconstruction error / MIG error for Beta VAE with different beta values

BETA_VALUES="1 2 4 8 16 32 64"

for beta_value in $BETA_VALUES
do
  echo "Running Beta VAE (Beta = $beta_value)..."
  python train.py --model_type beta_vae --seed 1234 --beta $beta_value
done
