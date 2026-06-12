# U-Net for Semantic Segmentation

This project implements and trains a U-Net-style semantic segmentation model on
ADE20K, a 150-class natural scene parsing dataset.

Original paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), Ronneberger, Fischer, and Brox, 2015.

The implementation follows the main U-Net structure: an encoder compresses the
input image into lower-resolution semantic features, and a decoder upsamples
those features back into a dense per-pixel prediction map. Skip connections pass
same-scale encoder features into the decoder so spatial detail is preserved
while the model builds higher-level context.

## Implementation

Core files:

```text
unet_train_parts/model.py       U-Net-style model definition
unet_train_parts/dataloader.py  ADE20K dataset loader
unet_train_ade20k.py            training entrypoint
data/README.md                  expected ADE20K directory layout
```

This is not a literal reproduction of the 2015 biomedical U-Net. It keeps the
encoder-decoder shape and concatenated skip connections, but uses several modern
implementation choices:

- residual convolution blocks
- group normalization
- strided convolutions for downsampling
- transposed convolutions for upsampling
- same-padding convolutions

The training setup uses:

- per-pixel cross entropy
- `ignore_index=-1` for unlabeled ADE20K pixels
- AdamW
- cosine learning-rate scheduling with warmup
- Hugging Face Accelerate for multi-GPU training

## Data

The dataset is not included in this repository. The code expects the ADE20K
train/validation split in the following form:

```text
data/ADEChallengeData2016/
├── images/
│   ├── training/
│   └── validation/
└── annotations/
    ├── training/
    └── validation/
```

Use `--path_to_data` to point the training script at a different local dataset
location.

## Completed Run

The completed run trained from scratch on ADE20K with the following setup:

- image size: `256`
- epochs: `150`
- hardware: `4 x RTX 2080 Ti`
- dataset: ADE20K train/validation split

Final training-log metrics:

- train loss: `1.1882`
- train pixel accuracy: `64.24%`
- validation loss: `1.3005`
- validation pixel accuracy: `62.42%`

The saved best checkpoint was later re-evaluated with nearest-neighbor mask
resizing and standard ignored-pixel handling:

- validation pixel accuracy, excluding ignored pixels: `69.63%`
- validation pixel accuracy, including ignored pixels: `63.79%`
- mean pixel accuracy: `26.41%`
- mean IoU: `19.77%`

## Interpretation

The mIoU result is the stricter metric and should be treated as the primary
quality signal. The model learns useful common-class segmentation behavior, but
it is not competitive with modern ADE20K systems that use pretrained backbones,
multi-scale context modules, and stronger decoder heads.

This run is best understood as a from-scratch educational implementation of a
U-Net-style segmentation model on a substantially harder task than the original
biomedical setting.
