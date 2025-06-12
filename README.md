# Disentanglement with Beta VAE and Factor VAE

This project provides a PyTorch Lightning implementation for training and evaluating two popular models for learning disentangled representations: β-VAE and Factor-VAE. The goal is to explore how these models encourage the separation of underlying generative factors in data, using standard datasets like dSprites and MPI3D.


## Project
The objective of this project is to implement and compare models that learn disentangled representations. A disentangled representation is one where single latent units are sensitive to changes in single generative factors, while being relatively invariant to changes in other factors.

This is achieved by modifying the standard VAE objective to penalize the model for learning entangled representations. This project focuses on:

- β-VAE: Encourages disentanglement by placing a stronger constraint on the latent channel's capacity, forcing it to learn an efficient, factorized representation. [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/pdf?id=Sy2fzU9gl)
- Factor-VAE: Employs an adversarial approach by adding a discriminator that penalizes the model if the dimensions of its latent code are not statistically independent. [Disentangling by Factorising](https://arxiv.org/pdf/1802.05983)

The models are evaluated using standard disentanglement metrics like the Mutual Information Gap (MIG) [Isolating Sources of Disentanglement in VAEs](https://arxiv.org/pdf/1802.04942).

Note: The dataset used are automaically downloaded in the train test phase. To download them refers to the URLs present in the [config.py](https://arxiv.org/pdf/1802.04942) file.

## Requirements
These are the requirements for the project:
```
python=3.11
torch
pytorch_lightning
torchvision
matplotlibå
scikit-learn
tensorboard
requests
```

For a quick env (miniconda) e library installation run:
```
sh install.sh
```

## How to Train
To train the models the [train.py](https://arxiv.org/pdf/1802.04942) implementation can be used with the default parameters provided in the [config.py](https://arxiv.org/pdf/1802.04942) or by overriding them:

```
python train.py --model_type beta_vae --dataset dsprites --beta 4.0 --seed 42 --epochs 50
```

### Arguments

- <b>--model_type:</b> [beta_vae, factor_vae]
- <b>--dataset:</b> [dsprites, mpi3d]
- <b>--beta:</b> hyperparameter for the beta vae model (default = 1)
- <b>--gamma:</b> hyperparameter for the factor vae model (default = 1)

### Reproducibility
To reproduce the same results of the project (without changing the default.py file) run:
```
sh run_train.sh
```

### Train results

The results of the training step are saved in these folders:
- [checkpoints](https://arxiv.org/pdf/1802.04942): best models obtained with different values for beta and gamma on the datasets dsprites and mpi3d
- [logs](https://arxiv.org/pdf/1802.04942): the logs of each run with the most important tracked values using tensorboard

## How to test
To evaluate the models the [test.py](https://arxiv.org/pdf/1802.04942) implementation can be used with the default parameters provided in the [config.py](https://arxiv.org/pdf/1802.04942) or by overriding them. The hyperparameters are saved in the training phase in the ckpt.

```
python evaluate.py --model_type beta_vae --dataset dsprites --checkpoint "path to the ckpt file"
```

### Reproducibility
To reproduce the same results of the project (without changing the default.py file) run:
```
sh run_test.sh
```

## Results

Beta VAE shows a clear reconstruction–disentanglement trade-off: as beta increases, reconstruction error degrades, while disentanglement improves.

Factor VAE achieves a better balance by penalizing Total Correlation (TC). It maintains a stable reconstruction and reaches a MIG value up to 0.22 on dSprites and 0.21 on MPI3D. The increasing confusion of the discriminator confirms improved latent independence with higher γ.

On the complex MPI3D dataset, both models achieved the highest MIG value with low beta or gamma values. This suggests that richer data may inherently support better factor separation, even with weaker regularization.

### dSprites

#### Beta VAE

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png