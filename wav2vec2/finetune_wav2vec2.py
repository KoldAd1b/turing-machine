"""
Most of this code was heavily inspired by the FineTune Wav2Vec2 Example Provided by 🤗 Huggingface!!!
https://huggingface.co/blog/fine-tune-wav2vec2-english

"""

import os
os.environ.setdefault("WANDB_PROJECT", "Wav2Vec2_LibriSpeech100_Finetune")
os.environ.setdefault("WANDB_MODE", "offline")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import jiwer
import argparse 

from transformers import (
    Wav2Vec2CTCTokenizer, 
    Trainer, 
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor
)

from dataset import LibriSpeechDataset, Wav2Vec2CollateFunctionForCTC
from utils import Wav2Vec2Config
from model import Wav2Vec2ForCTC

import warnings
warnings.filterwarnings("ignore")

def parse_arguments():

    parser = argparse.ArgumentParser(description="Wav2Vec2 Finetuning Arguments on Librispeech")

    ### Experiment Logging ###
    parser.add_argument(
        "--experiment_name", 
        required=True, 
        type=str
    )
    
    parser.add_argument(
        "--working_directory",
        required=True, 
        type=str
    )
    
    parser.add_argument(
        "--num_train_epochs",
        help="Number of epochs you want to train for",
        default=30, 
        type=int 
    )

    parser.add_argument(
        "--save_steps", 
        help="After how many steps do you want to log a checkpoint",
        default=500, 
        type=int
    )

    parser.add_argument(
        "--eval_steps", 
        help="After how many steps do you want to evaluate on eval data",
        default=500, 
        type=int
    )

    parser.add_argument(
        "--logging_steps", 
        help="After how many steps do you want to log to Weights and Biases (if installed)",
        default=500, 
        type=int
    )

    parser.add_argument(
        "--warmup_steps", 
        help="Number of learning rate warmup steps",
        default=1000, 
        type=int
    )

    ### Dataset Arguments ###
    parser.add_argument(
        "--path_to_dataset_root",
        help="Path to Librispeech dataset",
        required=True, 
        type=str
    )

    parser.add_argument(
        "--train_splits", 
        help="Select Librispeech Training Splits (960 Hours of data Split into 100, 360 and 500 hour sections)",
        default=["train-clean-100", "train-clean-360", "train-other-500"],
        choices=("train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "test-clean"),
        nargs='+',
        type=str
    )

    parser.add_argument(
        "--test_splits",
        help="Select Librispeech Testing Splits (Select from limited Validation + Testing Datasets)",
        default=["dev-clean", "test-clean"],
        choices=("train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "test-clean"),
        nargs="+",
        type=str
    )

    parser.add_argument(
        "--bad_audio_file",
        help="Path to a list of known-bad/corrupt audio files (tab-separated, path in first column) to skip during dataset indexing",
        default=None,
        type=str
    )

    ### Training Arguments ###

    parser.add_argument(
        "--per_device_batch_size",
        help="Batch size for every gradient accumulation steps",
        default=64, 
        type=int
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        help="Number of gradient accumulation steps you want",
        default=1, 
        type=int
    )

    parser.add_argument(
        "--learning_rate", 
        help="Max learning rate that we warmup to",
        default=1e-4, 
        type=float
    )

    parser.add_argument(
        "--weight_decay", 
        help="Weight decay applied to model parameters during training",
        default=0.005, 
        type=float
    )

    parser.add_argument(
        "--save_total_limit", 
        help="Max number of checkpoints to save",
        default=4, 
        type=int
    )

    parser.add_argument(
        "--group_by_length", 
        help="Huggingface trainer will automatically sort by length so less padding is used.\
             This will add some extra time at the start for the model to iterate through the data",
        action=argparse.BooleanOptionalAction,
        default=False
    )

    parser.add_argument(
        "--gradient_checkpointing", 
        help="If you want to enable gradient checkpointing in trainer",
        action=argparse.BooleanOptionalAction,
        default=False
    )

    parser.add_argument(
        "--mixed_precision",
        help="Mixed precision mode. Use bf16 on newer GPUs, fp16 on older cards, or set explicit mode.",
        default="auto",
        choices=("auto", "fp16", "bf16", "no")
    )
    
    ### Backbone Arguments ###
    parser.add_argument(
        "--huggingface_model_name",
        help="Name for pretrained Wav2Vec2Model backbone and Tokenizer",
        default="facebook/wav2vec2-base",
        type=str
    )

    parser.add_argument(
        "--path_to_pretrained_backbone",
        help="Path to model weights stored from our pretraining to initialize the backbone",
        default=None,
        type=str
    )

    parser.add_argument(
        "--pretrained_backbone",
        help="Do you want want a `pretrained` backbone that we made (need to provide path_to_pretrained_backbone), \
            `pretrained_huggingface` backbone (then need huggingface_model_name), or `random` initialized backbone",
        choices=("pretrained", "pretrained_huggingface", "random"),
        type=str
    )

    parser.add_argument(
        "--freeze_feature_extractor",
        help="Flag for if we want to freeze gradient updates on convolutional feature extractor",
        action=argparse.BooleanOptionalAction,
        default=False
    )  

    ### Prediction Head Arguments ###
    parser.add_argument(
        "--asr_head_dropout_p",
        help="Dropout probability on asr head",
        default=0.1, 
        type=float
    )

    args = parser.parse_args()
    
    return args


def resolve_mixed_precision(mixed_precision: str):
    if mixed_precision == "no":
        return None

    if mixed_precision == "auto":
        if not torch.cuda.is_available():
            return None

        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "bf16"
        return "fp16"

    return mixed_precision

### Load Arguments ###
args = parse_arguments()
resolved_mixed_precision = resolve_mixed_precision(args.mixed_precision)

### Load Tokenizer/Feature Extractor/ Processor ###
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.huggingface_model_name)
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

### Load Collate function ###
data_collator = Wav2Vec2CollateFunctionForCTC(processor=processor)

### Load Config ###
### Makes sure backbone portions of config identical to pretrained model! ###
### Or we wont be able to load our pretrained weights in ###
config = Wav2Vec2Config(hf_model_name=args.huggingface_model_name, 
                        blank_token_idx=tokenizer.pad_token_id,
                        vocab_size=tokenizer.vocab_size,
                        asr_head_dropout_p=args.asr_head_dropout_p,
                        path_to_pretrained_weights=args.path_to_pretrained_backbone,
                        pretrained_backbone=args.pretrained_backbone,
                        mlp_dropout_p=0.1, 
                        attention_dropout_p=0.1,
                        transformer_encoder_dropout=0.1, 
                        )

### Load Model ###
model = Wav2Vec2ForCTC(config)
if args.freeze_feature_extractor:
    model.freeze_feature_extractor()

### The HF backbone enables reentrant gradient checkpointing by default, which is
### incompatible with Accelerate/DDP ("marked a variable ready twice"). We have memory
### headroom at this batch size, so disable it and let Accelerate's standard DDP handle
### the plain model. (No-op for our custom backbone, which has no checkpointing.) ###
if getattr(model.wav2vec2, "is_gradient_checkpointing", False):
    model.wav2vec2.gradient_checkpointing_disable()

### Compute Word Error Rate ###
def compute_metrics(pred):

    ### As per the Huggingface Tutorial for Evaluation (https://huggingface.co/learn/nlp-course/chapter3/3) ###
    ### compute metrics is passed pred.predictions amd the true labels pred.label_ids ###
    ### We can decode both of these and compute the word error rate between them ###

    ### Grab raw logit predictions, and argmax to get predicted token ###
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    ### Convert all trailing pad tokens (indicated by -100) back to the original pad token id ###
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    ### Convert preds and targets back to strings ###
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    ### Compute Word Error Rate (jiwer aggregates over the full list of utterances) ###
    wer = jiwer.wer(reference=label_str, hypothesis=pred_str)

    return {"wer": wer}

### Define Training Arguments ###
training_args = TrainingArguments(
  output_dir=os.path.join(args.working_directory, args.experiment_name),
  group_by_length=args.group_by_length,
  per_device_train_batch_size=args.per_device_batch_size,
  gradient_accumulation_steps=args.gradient_accumulation_steps,
  eval_strategy="steps",
  num_train_epochs=args.num_train_epochs,
  save_steps=args.save_steps,
  eval_steps=args.eval_steps,
  logging_steps=args.logging_steps,
  learning_rate=args.learning_rate,
  weight_decay=args.weight_decay,
  warmup_steps=args.warmup_steps,
  fp16=resolved_mixed_precision == "fp16",
  bf16=resolved_mixed_precision == "bf16",
  save_total_limit=args.save_total_limit,
  run_name=args.experiment_name
)

### Load Training and Testing Datasets ###
trainset = LibriSpeechDataset(args.path_to_dataset_root,
                              include_splits=args.train_splits,
                              max_audio_duration=15,
                              truncate_audio=False,
                              return_transcripts=True,
                              bad_audio_file=args.bad_audio_file,
                              hf_model_name=args.huggingface_model_name)

testset = LibriSpeechDataset(args.path_to_dataset_root,
                             include_splits=args.test_splits,
                             max_audio_duration=15,
                             truncate_audio=False,
                             return_transcripts=True,
                             bad_audio_file=args.bad_audio_file,
                             hf_model_name=args.huggingface_model_name)

### Define Trainer ###
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=trainset,
    eval_dataset=testset,
    tokenizer=processor,
)

### Train Model! ###
trainer.train()

### Save Final Model ###
trainer.save_model()
    
