#!/bin/bash

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda create -y -n dis python=3.11
conda activate dis
pip install torch
pip install pytorch_lightning
pip install requests
pip install torchvision
pip install matplotlib
pip install scikit-learn
pip install tensorboard