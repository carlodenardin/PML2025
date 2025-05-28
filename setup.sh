#!/bin/bash

ENV_NAME="anomaly-detection"

source ~/miniconda3/etc/profile.d/conda.sh

if ! conda env list | grep -q "^$ENV_NAME\s"; then
    echo "Creating $ENV_NAME..."
    conda env create -f environment.yml
fi

echo "Activating $ENV_NAME..."
conda activate $ENV_NAME