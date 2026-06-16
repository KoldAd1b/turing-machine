# HiFiGAN on LJSpeech

This project implements and trains HiFiGAN, a neural vocoder that converts mel
spectrograms into waveform audio.

HiFiGAN is trained adversarially: a generator synthesizes waveform segments from
mel features, while discriminator networks learn to distinguish generated audio
from real speech. The training objective combines adversarial loss, feature
matching loss, and mel reconstruction loss.

## Implementation

Core files:

```text
model.py      HiFiGAN generator and discriminator modules
dataset.py    LJSpeech mel/audio dataset and audio-mel conversion utilities
loss.py       adversarial and feature-matching losses
train.py      training entrypoint
train.sh      completed-run launch configuration
```

The implementation includes:

- transposed-convolution upsampling in the generator
- multi-receptive-field residual blocks
- multi-period discriminator
- multi-scale discriminator
- mel spectrogram reconstruction loss
- feature matching loss
- adversarial generator and discriminator losses

## Data

The dataset is not included in this repository. The completed run used
LJSpeech with train and validation manifest CSV files.

Expected local inputs:

```text
data/LJSpeech-1.1/
├── wavs/
├── train_metadata.csv
└── test_metadata.csv
```

The model was trained from ground-truth audio/mel pairs rather than from
Tacotron2-predicted mels.

## Completed Run

The completed run trained from scratch on LJSpeech with the following setup:

- training steps: `150,000`
- mixed precision: `fp16`
- hardware: `4 x RTX 2080 Ti`
- batch size: `24` per GPU
- global batch size: `96`
- learning rate: `0.0002`
- sampling rate: `22,050 Hz`
- mel channels: `80`
- segment size: `8,192`
- hop size: `256`
- checkpoint interval: `10,000` steps
- eval interval: `5,000` steps

Final metrics:

- final train mel loss: `0.1655`
- final validation mel loss: `0.1655`
- best validation mel loss: `0.1655` at step `150,000`
- final generator loss: `22.2834`
- final discriminator loss: `2.7632`

The validation mel loss continued improving through the end of training, with
the best validation result occurring at the final step.

## Visualizations and Samples

Generated artifacts are stored in:

```text
visualizations/
samples/
```

Included visualizations:

- `training_curves.png`
- `validation_mel_loss.png`
- `metric_summary.png`
- `01_LJ001-0023_spectrogram_comparison.png`
- `04_LJ003-0004_spectrogram_comparison.png`
- `06_LJ003-0082_spectrogram_comparison.png`

Included audio samples:

- six generated/reference `.wav` pairs
- `samples/manifest.csv`
- `samples/README.md`

The sample pairs were generated from validation-set ground-truth mels using the
final checkpoint. They are intended for direct listening comparison between
reference LJSpeech audio and HiFiGAN-generated audio.

## Interpretation

This run demonstrates a working from-scratch neural vocoder. Mel loss provides
a useful training signal, but final quality should be judged by listening to
the generated/reference sample pairs. The adversarial objective is intended to
recover sharper waveform detail than a reconstruction-only audio model.
