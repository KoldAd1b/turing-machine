# U-Net for Semantic Segmentation

I used to think image models mostly answered one question:

what is in this image?

Then segmentation forces a much sharper question:

what is every single pixel?

That shift is the whole game. Classification gives one label. Segmentation gives a dense map. Same image, but now the model has to preserve meaning and location at the same time. Annoying, right? Also genuinely beautiful.

This project implements a U-Net-style model for semantic segmentation and trains it on ADE20K.

Original paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), Ronneberger, Fischer, and Brox, 2015.

## The problem is precision

For a normal image classifier, the output is simple:

```text
image -> class label
```

For segmentation, the output has to stay spatial:

```text
image:  [3, H, W]
output: [num_classes, H, W]
```

Every pixel gets class logits.

So the model needs two things that fight each other a little:

- context, because a pixel only makes sense inside the larger scene
- precision, because the final answer still has to land on exact pixels

This is the tension U-Net solves so cleanly.

## The beautiful move

U-Net has a contracting path and an expanding path.

The encoder compresses the image:

```text
high resolution, low-level detail
        |
        v
lower resolution, higher-level context
```

The decoder expands it back:

```text
compressed context
        |
        v
pixel-level prediction
```

If that were the whole architecture, the model would have a problem. The bottleneck knows what is in the image, but a lot of exact boundary information has been squeezed away.

So U-Net keeps the encoder features and hands them back to the decoder through skip connections:

```text
encoder feature map ----------------------+
                                          |
                                          v
upsampled decoder feature -> concatenate -> conv block
```

That is the part I find hard to get over.

The model goes down to understand the scene, then comes back up with the details it saved along the way. Context and precision, both carried through the network. Simple, and a little bit magic.

## What is in this implementation

The model lives here:

```text
unet_train_parts/model.py
```

This version is not a literal line-by-line copy of the 2015 paper. It keeps the U-Net shape, but uses a few modern choices:

- residual convolution blocks
- group normalization
- strided convolutions for downsampling
- transposed convolutions for upsampling
- concatenated skip connections between encoder and decoder stages

The ADE20K dataset code lives here:

```text
unet_train_parts/dataloader.py
```

The training entrypoint lives here:

```text
unet_train_ade20k.py
```

The training setup uses:

- per-pixel cross entropy
- `ignore_index=-1` for unlabeled pixels
- AdamW
- cosine learning-rate schedule with warmup
- Hugging Face Accelerate for multi-GPU training

## The mental model I keep

The encoder asks:

what am I looking at?

The decoder asks:

where exactly is it?

The skip connections make sure the second question does not have to be answered from memory alone.

That is the principle I like here: compression is powerful, but recovery needs receipts. U-Net keeps those receipts.
