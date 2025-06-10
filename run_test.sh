#!/bin/bash

MODEL_TYPE="beta_vae"
CHECKPOINTS_DIR="checkpoints/dsprites/$MODEL_TYPE"

for experiment_dir in "$CHECKPOINTS_DIR"/*
do
    checkpoint_file=$(find "$experiment_dir" -name "*.ckpt" -print -quit)   
    python test.py --model_type $MODEL_TYPE --checkpoint "$checkpoint_file"     
done


MODEL_TYPE="factor_vae"
CHECKPOINTS_DIR="checkpoints/dsprites/$MODEL_TYPE"

for experiment_dir in "$CHECKPOINTS_DIR"/*
do
    checkpoint_file=$(find "$experiment_dir" -name "*.ckpt" -print -quit)   
    python test.py --model_type $MODEL_TYPE --checkpoint "$checkpoint_file"     
done
