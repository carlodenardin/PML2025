#!/bin/bash

# DSPRITES - BETA VAE
MODEL_TYPE="beta_vae"
CHECKPOINTS_DIR="checkpoints/dsprites/$MODEL_TYPE"

for experiment_dir in "$CHECKPOINTS_DIR"/*
do
    checkpoint_file=$(find "$experiment_dir" -name "*.ckpt" -print -quit)   
    python test.py --model_type $MODEL_TYPE --checkpoint "$checkpoint_file"     
done

# DSPRITES - FACTOR VAE
MODEL_TYPE="factor_vae"
CHECKPOINTS_DIR="checkpoints/dsprites/$MODEL_TYPE"

for experiment_dir in "$CHECKPOINTS_DIR"/*
do
    checkpoint_file=$(find "$experiment_dir" -name "*.ckpt" -print -quit)   
    python test.py --model_type $MODEL_TYPE --checkpoint "$checkpoint_file"     
done

# MPI3D - BETA VAE
MODEL_TYPE="beta_vae"
CHECKPOINTS_DIR="checkpoints/mpi3d/$MODEL_TYPE"

for experiment_dir in "$CHECKPOINTS_DIR"/*
do
    checkpoint_file=$(find "$experiment_dir" -name "*.ckpt" -print -quit)   
    python test.py --model_type $MODEL_TYPE --dataset mpi3d --checkpoint "$checkpoint_file"     
done

# MPI3D - FACTOR VAE
MODEL_TYPE="factor_vae"
CHECKPOINTS_DIR="checkpoints/mpi3d/$MODEL_TYPE"

for experiment_dir in "$CHECKPOINTS_DIR"/*
do
    checkpoint_file=$(find "$experiment_dir" -name "*.ckpt" -print -quit)   
    python test.py --model_type $MODEL_TYPE --dataset mpi3d --checkpoint "$checkpoint_file"     
done