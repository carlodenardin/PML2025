#!/bin/bash

# DSPRITES - BETA VAE
BETA_VALUES="1 2 4 8 16 32 64"

for beta_value in $BETA_VALUES
do
  echo "Running Beta VAE (Beta = $beta_value)..."
  python train.py --model_type beta_vae --seed 19 --beta $beta_value
done

# DSPRITES - FACTOR VAE
GAMMA_VALUES="1 2 4 8 16 32 64"

for gamma_value in $GAMMA_VALUES
do
  echo "Running Factor VAE (Gamma = $gamma_value)..."
  python train.py --model_type factor_vae --seed 19 --gamma $gamma_value
done

# MPI3D - BETA VAE
BETA_VALUES="1 2 4 8 16 32 64"

for beta_value in $BETA_VALUES
do
  echo "Running Beta VAE (Beta = $beta_value)..."
  python train.py --model_type beta_vae --seed 19 --beta $beta_value --dataset mpi3d
done

# MPI3D - FACTOR VAE

GAMMA_VALUES="1 2 4 8 16 32 64"

for gamma_value in $GAMMA_VALUES
do
  echo "Running Factor VAE (Gamma = $gamma_value)..."
  python train.py --model_type factor_vae --seed 19 --gamma $gamma_value --dataset mpi3d
done

