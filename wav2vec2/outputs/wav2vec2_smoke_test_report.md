# Wav2Vec2 Smoke Test + Validation Artifacts

Source run: `learning-deep/audio/Wav2Vec2/work_dir/finetune_from_scratch`

## Test / Validation Checks

- Loaded `trainer_state.json` from `checkpoint-15850` and parsed `log_history` as structured source of truth.
- Parsed unified history rows: `3233`.
- Train rows: `3170`.
- Evaluation rows: `63`.
- Confirmed run log exists and contains `9513` dict-like lines with metrics.

## Summary Metrics

- Final train step: `15850`
- Final train loss (logged): `0.0036`
- Final epoch: `49.84409448818898`
- Final learning rate: `3.367003367003367e-08`

### Eval
- Best eval loss: `0.9421365261077881` at step `3250`
- Best eval WER: `0.6608306188925082` at step `15750`
- Final eval loss: `2.550819158554077` at step `15750`

### Output artifacts
- CSV: `outputs/wav2vec2_eval_log_parsed.csv`
- SVG charts: `visualizations/wav2vec2_train_loss.svg`, `visualizations/wav2vec2_eval_loss.svg`, `visualizations/wav2vec2_eval_wer.svg`, `visualizations/wav2vec2_learning_rate.svg`, `visualizations/wav2vec2_eval_loss_and_wer.svg`

### Environment smoke test (with `conda activate deep`)

- Executed in `turing-machine/wav2vec2` using the completed checkpoint:
  - `learning-deep/audio/Wav2Vec2/work_dir/finetune_from_scratch/checkpoint-15850/model.safetensors`
- Loaded custom `Wav2Vec2Config` with:
  - `pretrained_backbone='pretrained'`
  - `path_to_pretrained_weights=<checkpoint>/model.safetensors`
  - `vocab_size=32`, `blank_token_idx=0`
- Loaded tokenizer from `learning-deep/audio/Wav2Vec2/work_dir/finetune_from_scratch`
- Performed forward passes with synthetic 16 kHz audio tensors.

Smoke results:
- Forward forward pass: logits produced without errors: `(1, 59, 32)` for a 1.2s sine signal.
- CTC forward with labels: loss tensor computed successfully (`26.25345` for synthetic batch).
- CUDA visibility check: available (`torch.cuda.is_available() = True`) with 4 devices.
- Device smoke pass: model forward succeeds on `cuda:0`, logits shape `(1, 19, 32)`.

To run the same check locally:

```bash
cd /home/axa220303/Desktop/turing-machine/wav2vec2
source /home/axa220303/miniforge3/etc/profile.d/conda.sh
conda activate deep
python - <<'PY'
from pathlib import Path
import sys
import torch
import numpy as np
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from safetensors.torch import load_file

sys.path.append('/home/axa220303/Desktop/turing-machine/wav2vec2')
from utils import Wav2Vec2Config
from model import Wav2Vec2ForCTC

ckpt_dir = Path('/home/axa220303/Desktop/learning-deep/audio/Wav2Vec2/work_dir/finetune_from_scratch/checkpoint-15850')
run_root = Path('/home/axa220303/Desktop/learning-deep/audio/Wav2Vec2/work_dir/finetune_from_scratch')

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(run_root.as_posix())
processor = Wav2Vec2Processor(
  feature_extractor=Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False),
  tokenizer=tokenizer,
)

sr = 16000
time = torch.arange(int(1.2 * sr)).float() / sr
wave = (0.1 * torch.sin(2 * np.pi * 220.0 * time).numpy()).astype(np.float32)

cfg = Wav2Vec2Config(pretrained_backbone='pretrained',
                     path_to_pretrained_weights=str(ckpt_dir / 'model.safetensors'),
                     vocab_size=tokenizer.vocab_size,
                     blank_token_idx=tokenizer.pad_token_id)
model = Wav2Vec2ForCTC(cfg)
model.load_state_dict(load_file(str(ckpt_dir / 'model.safetensors')), strict=False)

with torch.no_grad():
    proc = processor(wave, sampling_rate=sr, return_tensors='pt', padding=True)
    _, logits = model(input_values=proc['input_values'])
    print('ok', tuple(logits.shape), tuple(torch.argmax(logits, dim=-1).shape))
PY
```
