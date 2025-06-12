# Disentanglement with Beta VAE and Factor VAE

This project provides a PyTorch Lightning implementation for training and evaluating two popular models for learning disentangled representations: β-VAE and Factor-VAE. The goal is to explore how these models encourage the separation of underlying generative factors in data, using standard datasets like dSprites and MPI3D.


## Project
The objective of this project is to implement and compare models that learn disentangled representations. A disentangled representation is one where single latent units are sensitive to changes in single generative factors, while being relatively invariant to changes in other factors.

This is achieved by modifying the standard VAE objective to penalize the model for learning entangled representations. This project focuses on:

- β-VAE: Encourages disentanglement by placing a stronger constraint on the latent channel's capacity, forcing it to learn an efficient, factorized representation. [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/pdf?id=Sy2fzU9gl)
- Factor-VAE: Employs an adversarial approach by adding a discriminator that penalizes the model if the dimensions of its latent code are not statistically independent. [Disentangling by Factorising](https://arxiv.org/pdf/1802.05983)

The models are evaluated using standard disentanglement metrics like the Mutual Information Gap (MIG) [Isolating Sources of Disentanglement in VAEs](https://arxiv.org/pdf/1802.04942).

Note: The dataset used are automaically downloaded in the train test phase. To download them refers to the URLs present in the [config.py](config.py) file.

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
To train the models the [train.py](train.py) implementation can be used with the default parameters provided in the [config.py](config.py) or by overriding them:

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
- [checkpoints](checkpoints/): best models obtained with different values for beta and gamma on the datasets dsprites and mpi3d
- [logs](logs/): the logs of each run with the most important tracked values using tensorboard

To view and interact with the results run this command:
```
tensorboard --logdir=logs/
```

## How to test
To evaluate the models the [test.py](test.py) implementation can be used with the default parameters provided in the [config.py](config.py) or by overriding them. The hyperparameters are saved in the training phase in the ckpt.

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

### dSprites dataset

#### Beta VAE ([reconstruction example](results/dsprites/beta_vae/reconstruction.png), [reconstruction loss](results/dsprites/beta_vae/reconstruction_loss.png))

| Beta | Rec Error | KL    |
|------|-----------|-------|
| 1    | 24.71     | 27.29 |
| 2    | 32.91     | 18.83 |
| 4    | 71.91     | 9.92  |
| 8    | 74.69     | 9.74  |
| 16   | 114.16    | 6.50  |
| 32   | 161.15    | 4.45  |
| 64   | 251.20    | 2.71  |

Table 01: validation loss

#### Factor VAE ([reconstruction example](results/dsprites/factor_vae/reconstruction.png), [reconstruction loss](results/dsprites/factor_vae/reconstruction_loss.png))

| Gamma | Rec Error | KL    | TC    | D(z_real.) | D(z_perm.) |
|-------|-----------|-------|-------|-------------|-------------|
| 1     | 29.23     | 24.32 | 5.36  | 0.96        | 0.04        |
| 2     | 39.17     | 20.42 | 2.57  | 0.85        | 0.16        |
| 4     | 46.30     | 17.60 | 1.12  | 0.69        | 0.28        |
| 8     | 48.88     | 17.86 | 0.70  | 0.64        | 0.33        |
| 16    | 50.11     | 15.82 | 0.39  | 0.58        | 0.41        |
| 32    | 58.58     | 16.27 | 0.08  | 0.52        | 0.48        |
| 64    | 56.28     | 16.29 | -0.20 | 0.45        | 0.43        |

Table 02: validation loss and discriminator prediction

### MPI3D dataset

#### Beta VAE ([reconstruction example](results/mpi3d/beta_vae/reconstruction.png), [reconstruction loss](results/mpi3d/beta_vae/reconstruction_loss.png))

| Beta | Rec Error | KL    |
|------|-----------|-------|
| 1    | 11.1      | 10.34 |
| 2    | 16.65     | 6.88  |
| 4    | 28.90     | 3.69  |
| 8    | 76.36     | 9.64  |
| 16   | 53.72     | 1.12  |
| 32   | 71.88     | 0.41  |
| 64   | 87.03     | 0.01  |

Table 03: validation loss

#### Factor VAE ([reconstruction example](results/mpi3d/factor_vae/reconstruction.png), [reconstruction loss](results/mpi3d/factor_vae/reconstruction_loss.png))

| Gamma | Rec Error | KL   | TC    | D(z_real.) | D(z_perm.) |
|-------|-----------|------|-------|-------------|-------------|
| 1     | 14.91     | 9.12 | 0.3   | 0.55        | 0.43        |
| 2     | 12.94     | 9.32 | 0.23  | 0.54        | 0.47        |
| 4     | 14.01     | 9.25 | 0.23  | 0.54        | 0.47        |
| 8     | 13.81     | 9.21 | 0.09  | 0.52        | 0.49        |
| 16    | 16.68     | 8.27 | 0.001 | 0.48        | 0.48        |
| 32    | 16.24     | 8.47 | 0.03  | 0.5         | 0.5         |
| 64    | 28.18     | 5.92 | 0.02  | 0.5         | 0.5         |

Table 04: validation loss and discriminator prediction