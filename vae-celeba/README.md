# VAE on CelebA

This project implements and trains a convolutional variational autoencoder
(VAE) on CelebA face images.

The model learns a compressed latent representation of each image and decodes
samples from that latent space back into image space. The training objective
combines pixel reconstruction loss with the KL-divergence regularization term
used in the standard VAE formulation.

## Implementation

Core files:

```text
model.py       convolutional VAE model definition
train_vae.py   training entrypoint
```

The model uses:

- convolutional encoder and decoder blocks
- residual convolution blocks
- group normalization
- spatial latent tensors with 8 latent channels
- reparameterized Gaussian latent sampling
- sigmoid output mapping to image space

The training setup uses:

- CelebA images resized to `128 x 128`
- mean-squared reconstruction loss summed over pixels
- KL-divergence latent regularization
- AdamW
- cosine learning-rate scheduling with warmup
- Hugging Face Accelerate for multi-GPU training

## Data

The dataset is not included in this repository. The original run used CelebA
aligned face images arranged for `torchvision.datasets.ImageFolder`.

Expected local layout:

```text
data/celeba/
└── img_align_celeba/
    ├── 000001.jpg
    ├── 000002.jpg
    └── ...
```

Use `LD_DATA_DIR` to point the training script at a different local dataset
location.

## Completed Run

The completed run trained on CelebA with the following setup:

- training iterations: `150,000`
- approximate epochs: `111.5`
- image size: `128 x 128`
- global batch size: `128`
- learning rate: `0.0005`
- warmup steps: `2,500`
- train split size: `172,209`
- eval split size: `30,390`

Final and best metrics:

- final train loss: `451.6373`
- final eval loss: `457.6924`
- best eval loss: `443.0262`
- best eval iteration: `112,500`

The best eval loss occurred before the final iteration, so the run appears to
have reached its useful training point before the end. Additional training alone
would likely not solve the main visual limitation: plain VAEs trained with pixel
reconstruction loss tend to produce smooth or blurry samples.

## Visualizations

Generated visuals are stored in:

```text
visualizations/
```

Included files:

- `loss_curves.png`
- `metric_summary.png`
- `reconstruction_pairs.png`
- `random_samples.png`
- `latent_interpolation.png`
- `best_vae_recon_grid.png`

The reconstruction examples show that the model learned face structure, pose,
skin tone, hair color, and broad lighting patterns. Fine identity details are
smoothed, which is expected for this VAE objective.
