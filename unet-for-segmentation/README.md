# U-Net for Semantic Segmentation

This project implements a U-Net-style model for semantic segmentation.

Paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), Ronneberger, Fischer, and Brox, 2015.

## What U-Net Does

Semantic segmentation is pixel-level classification. Instead of predicting one label for an entire image, the model predicts a class for every pixel.

For an RGB input image:

```text
image:  [3, H, W]
output: [num_classes, H, W]
```

Each pixel gets a class score. During training, cross entropy compares those per-pixel scores against the segmentation mask.

## Architecture

U-Net has two main paths:

```text
Input image
    |
    v
Encoder / contracting path
    - keeps increasing feature channels
    - keeps reducing spatial resolution
    - learns context
    |
    v
Bottleneck
    - lowest-resolution, highest-channel representation
    |
    v
Decoder / expanding path
    - upsamples spatial resolution
    - reduces feature channels
    - combines coarse context with fine detail
    |
    v
Per-pixel class logits
```

The important idea is the skip connection:

```text
encoder feature map  --------------------+
                                         |
                                         v
decoder feature map after upsampling -> concatenate -> convolution block
```

The encoder sees the image at progressively coarser scales, so it learns what is present. The decoder restores resolution, so it learns where things are. Skip connections carry high-resolution spatial detail from the encoder into the decoder.

That is what makes U-Net useful for segmentation: it does not force the model to reconstruct fine boundaries only from the compressed bottleneck.

## This Implementation

The model is defined in:

```text
unet_train_parts/model.py
```

It uses:

- residual convolution blocks
- group normalization
- strided convolutions for downsampling
- transposed convolutions for upsampling
- concatenated skip connections between encoder and decoder stages

The ADE20K dataset loader is defined in:

```text
unet_train_parts/dataloader.py
```

The training entrypoint is:

```text
unet_train_ade20k.py
```

It trains the model on ADE20K scene segmentation with:

- per-pixel cross entropy loss
- `ignore_index=-1` for unlabeled pixels
- AdamW optimization
- cosine learning-rate schedule with warmup
- Hugging Face Accelerate for multi-GPU training

## Mental Model

U-Net works because segmentation needs both:

- **semantic context**: what object or region is this?
- **spatial precision**: exactly which pixels belong to it?

The encoder is good at context. The decoder is good at resolution. The skip connections let the decoder recover boundaries and fine structure without losing the encoder's higher-level understanding.
