import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import set_seed

from dataset import BatchSampler, TTSCollator, TTSDataset, denormalize
from model import Tacotron2, Tacotron2Config
from tokenizer import Tokenizer


DEFAULT_WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work_dir")


def append_csv_row(path, fieldnames, row):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser()

    ### SETUP CONFIG ###
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--working_directory", type=str, default=DEFAULT_WORK_DIR)
    parser.add_argument("--save_audio_gen", type=str, default=None)
    parser.add_argument("--path_to_train_manifest", type=str, required=True)
    parser.add_argument("--path_to_val_manifest", type=str, required=True)
    parser.add_argument("--path_to_mels", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    ### TRAINING CONFIG ###
    parser.add_argument("--training_steps", type=int, default=25125)
    parser.add_argument("--console_out_iters", type=int, default=5)
    parser.add_argument("--wandb_log_iters", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=2500)
    parser.add_argument("--checkpoint_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_val_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    parser.add_argument("--start_decay_steps", type=int, default=None)

    ### MODEL CONFIG ###
    parser.add_argument("--character_embed_dim", type=int, default=512)
    parser.add_argument("--encoder_kernel_size", type=int, default=5)
    parser.add_argument("--encoder_n_convolutions", type=int, default=3)
    parser.add_argument("--encoder_embed_dim", type=int, default=512)
    parser.add_argument("--encoder_dropout_p", type=float, default=0.5)
    parser.add_argument("--decoder_rnn_embed_dim", type=int, default=1024)
    parser.add_argument("--decoder_dropout_p", type=float, default=0.1)
    parser.add_argument("--decoder_prenet_dim", type=int, default=256)
    parser.add_argument("--decoder_prenet_depth", type=int, default=2)
    parser.add_argument("--decoder_prenet_dropout_p", type=float, default=0.5)
    parser.add_argument("--decoder_postnet_num_convs", type=int, default=5)
    parser.add_argument("--decoder_postnet_n_filters", type=int, default=512)
    parser.add_argument("--decoder_postnet_kernel_size", type=int, default=5)
    parser.add_argument("--decoder_postnet_dropout_p", type=float, default=0.5)
    parser.add_argument("--attention_dim", type=int, default=128)
    parser.add_argument("--attention_dropout_p", type=float, default=0.1)
    parser.add_argument("--attention_location_n_filters", type=int, default=32)
    parser.add_argument("--attention_location_kernel_size", type=int, default=31)

    ### DATASET CONFIG ###
    parser.add_argument("--sampling_rate", type=int, default=22050)
    parser.add_argument("--num_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--hop_size", type=int, default=256)
    parser.add_argument("--min_db", type=float, default=-100.0)
    parser.add_argument("--max_scaled_abs", type=float, default=4.0)
    parser.add_argument("--fmin", type=int, default=0)
    parser.add_argument("--fmax", type=int, default=8000)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--log_wandb", action=argparse.BooleanOptionalAction)

    return parser.parse_args()


args = parse_args()

if args.seed is not None:
    set_seed(args.seed)

path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
if args.run_name is None:
    args.run_name = args.experiment_name
if args.save_audio_gen is None:
    args.save_audio_gen = os.path.join(path_to_experiment, "generated")

accelerator = Accelerator(project_dir=path_to_experiment,
                          log_with="wandb" if args.log_wandb else None)

if accelerator.is_main_process:
    os.makedirs(path_to_experiment, exist_ok=True)
    os.makedirs(args.save_audio_gen, exist_ok=True)

if args.log_wandb:
    accelerator.init_trackers(
        project_name=args.experiment_name,
        init_kwargs={"wandb": {"name": args.run_name}},
    )

accelerator.print(args)

train_log_path = os.path.join(path_to_experiment, "train_steps.csv")
eval_log_path = os.path.join(path_to_experiment, "eval_metrics.csv")
train_log_fields = [
    "step",
    "dataset_pass",
    "loss",
    "mel_loss",
    "refined_mel_loss",
    "stop_loss",
    "learning_rate",
]
eval_log_fields = [
    "step",
    "dataset_pass_estimate",
    "val_loss",
    "val_mel_loss",
    "val_refined_mel_loss",
    "val_stop_loss",
    "learning_rate",
]

tokenizer = Tokenizer()

config = Tacotron2Config(
    num_mels=args.num_mels,
    num_chars=tokenizer.vocab_size,
    character_embed_dim=args.character_embed_dim,
    pad_token_id=tokenizer.pad_token_id,
    encoder_kernel_size=args.encoder_kernel_size,
    encoder_n_convolutions=args.encoder_n_convolutions,
    encoder_embed_dim=args.encoder_embed_dim,
    encoder_dropout_p=args.encoder_dropout_p,
    decoder_embed_dim=args.decoder_rnn_embed_dim,
    decoder_dropout_p=args.decoder_dropout_p,
    decoder_prenet_dim=args.decoder_prenet_dim,
    decoder_prenet_depth=args.decoder_prenet_depth,
    decoder_prenet_dropout_p=args.decoder_prenet_dropout_p,
    decoder_postnet_num_convs=args.decoder_postnet_num_convs,
    decoder_postnet_n_filters=args.decoder_postnet_n_filters,
    decoder_postnet_kernel_size=args.decoder_postnet_kernel_size,
    decoder_postnet_dropout_p=args.decoder_postnet_dropout_p,
    attention_dim=args.attention_dim,
    attention_dropout_p=args.attention_dropout_p,
    attention_location_n_filters=args.attention_location_n_filters,
    attention_location_kernel_size=args.attention_location_kernel_size,
)

model = Tacotron2(config)
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
accelerator.print(f"Total Trainable Parameters: {total_trainable_params}")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    eps=args.adam_eps,
)

trainset = TTSDataset(
    args.path_to_train_manifest,
    sample_rate=args.sampling_rate,
    n_fft=args.n_fft,
    window_size=args.window_size,
    hop_size=args.hop_size,
    fmin=args.fmin,
    fmax=args.fmax,
    num_mels=args.num_mels,
    min_db=args.min_db,
    max_scaled_abs=args.max_scaled_abs,
    path_to_mels=args.path_to_mels,
)

testset = TTSDataset(
    args.path_to_val_manifest,
    sample_rate=args.sampling_rate,
    n_fft=args.n_fft,
    window_size=args.window_size,
    hop_size=args.hop_size,
    fmin=args.fmin,
    fmax=args.fmax,
    num_mels=args.num_mels,
    min_db=args.min_db,
    max_scaled_abs=args.max_scaled_abs,
    path_to_mels=args.path_to_mels,
)

collator = TTSCollator()
train_sampler = BatchSampler(
    trainset,
    batch_size=args.batch_size,
    drop_last=accelerator.num_processes > 1,
)

trainloader = DataLoader(
    trainset,
    batch_sampler=train_sampler,
    num_workers=args.num_workers,
    collate_fn=collator,
    pin_memory=True,
)

testloader = DataLoader(
    testset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=collator,
    pin_memory=True,
)

model, optimizer, trainloader, testloader = accelerator.prepare(
    model, optimizer, trainloader, testloader
)

steps_per_pass = len(trainloader)
using_scheduler = args.start_decay_steps is not None
if using_scheduler:
    decay_steps = max(args.training_steps - args.start_decay_steps, 1)
    min_lr_ratio = args.min_learning_rate / args.learning_rate

    def lr_lambda(step):
        if step < args.start_decay_steps:
            return 1.0
        return min_lr_ratio ** ((step - args.start_decay_steps) / decay_steps)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)
    accelerator.register_for_checkpointing(scheduler)
else:
    scheduler = None

if args.resume_from_checkpoint is not None:
    path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)
    with accelerator.main_process_first():
        accelerator.load_state(path_to_checkpoint)
    completed_steps = int(args.resume_from_checkpoint.split("_")[-1])
    completed_passes = completed_steps // steps_per_pass
    accelerator.print(f"Resuming from Step: {completed_steps}")
else:
    completed_steps = 0
    completed_passes = 0


def current_lr():
    return optimizer.param_groups[0]["lr"]


def save_eval_plot(mels, mels_postnet_out, attention_weights, step):
    true_mel = denormalize(
        mels[0].T.detach().float().to("cpu"),
        min_db=args.min_db,
        max_abs_val=args.max_scaled_abs,
    )
    pred_mel = denormalize(
        mels_postnet_out[0].T.detach().float().to("cpu"),
        min_db=args.min_db,
        max_abs_val=args.max_scaled_abs,
    )
    attention = attention_weights[0].T.detach().float().to("cpu")

    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    im0 = axes[0].imshow(true_mel, aspect="auto", origin="lower", interpolation="none")
    axes[0].set_title("True Mel")
    axes[0].set_ylabel("Mel bins")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred_mel, aspect="auto", origin="lower", interpolation="none")
    axes[1].set_title("Predicted Mel")
    axes[1].set_ylabel("Mel bins")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(attention, aspect="auto", origin="lower", interpolation="none")
    axes[2].set_title("Alignment")
    axes[2].set_ylabel("Character Index")
    axes[2].set_xlabel("Decoder Mel Timesteps")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_audio_gen, f"step_{step}_result.png"))
    plt.close(fig)


def compute_losses(mels_out, mels_postnet_out, stop_preds, mels, stops):
    mel_loss = F.mse_loss(mels_out.float(), mels.float())
    refined_mel_loss = F.mse_loss(mels_postnet_out.float(), mels.float())
    stop_loss = F.binary_cross_entropy_with_logits(
        stop_preds.reshape(-1, 1).float(),
        stops.reshape(-1, 1).float(),
    )
    loss = mel_loss + refined_mel_loss + stop_loss
    return loss, mel_loss, refined_mel_loss, stop_loss


def evaluate(step):
    accelerator.print("--VALIDATION--")
    model.eval()

    val_loss = 0
    val_mel_loss = 0
    val_refined_mel_loss = 0
    val_stop_loss = 0
    num_losses = 0
    saved_plot = False

    for val_step, (texts, text_lens, mels, stops, encoder_mask, decoder_mask) in enumerate(testloader):
        texts = texts.to(accelerator.device)
        mels = mels.to(accelerator.device)
        stops = stops.to(accelerator.device)
        encoder_mask = encoder_mask.to(accelerator.device)
        decoder_mask = decoder_mask.to(accelerator.device)

        with torch.no_grad():
            with accelerator.autocast():
                mels_out, mels_postnet_out, stop_preds, attention_weights = model(
                    texts, text_lens.to("cpu"), mels, encoder_mask, decoder_mask
                )

        loss, mel_loss, refined_mel_loss, stop_loss = compute_losses(
            mels_out, mels_postnet_out, stop_preds, mels, stops
        )

        val_loss += torch.mean(accelerator.gather_for_metrics(loss.detach()))
        val_mel_loss += torch.mean(accelerator.gather_for_metrics(mel_loss.detach()))
        val_refined_mel_loss += torch.mean(accelerator.gather_for_metrics(refined_mel_loss.detach()))
        val_stop_loss += torch.mean(accelerator.gather_for_metrics(stop_loss.detach()))
        num_losses += 1

        if accelerator.is_main_process and not saved_plot:
            save_eval_plot(mels, mels_postnet_out, attention_weights, step)
            saved_plot = True

        if args.max_val_steps is not None and val_step + 1 >= args.max_val_steps:
            break

    val_loss = val_loss.item() / max(num_losses, 1)
    val_mel_loss = val_mel_loss.item() / max(num_losses, 1)
    val_refined_mel_loss = val_refined_mel_loss.item() / max(num_losses, 1)
    val_stop_loss = val_stop_loss.item() / max(num_losses, 1)

    accelerator.print(
        "Validation Step {} | Loss {:.4f} | Mel Loss {:.4f} | RMel Loss {:.4f} | Stop Loss {:.4f}".format(
            step,
            val_loss,
            val_mel_loss,
            val_refined_mel_loss,
            val_stop_loss,
        )
    )

    if args.log_wandb:
        accelerator.log(
            {
                "val_mel_loss": val_mel_loss,
                "val_refined_mel_loss": val_refined_mel_loss,
                "val_stop_loss": val_stop_loss,
                "val_total_loss": val_loss,
            },
            step=step,
        )

    if accelerator.is_main_process:
        append_csv_row(
            eval_log_path,
            eval_log_fields,
            {
                "step": step,
                "dataset_pass_estimate": step / steps_per_pass,
                "val_loss": val_loss,
                "val_mel_loss": val_mel_loss,
                "val_refined_mel_loss": val_refined_mel_loss,
                "val_stop_loss": val_stop_loss,
                "learning_rate": current_lr(),
            },
        )

    model.train()


while completed_steps < args.training_steps:
    dataset_pass = completed_passes
    accelerator.print(f"Dataset Pass: {dataset_pass} | Starting Step: {completed_steps}")
    model.train()

    for texts, text_lens, mels, stops, encoder_mask, decoder_mask in trainloader:
        if completed_steps >= args.training_steps:
            break

        texts = texts.to(accelerator.device)
        mels = mels.to(accelerator.device)
        stops = stops.to(accelerator.device)
        encoder_mask = encoder_mask.to(accelerator.device)
        decoder_mask = decoder_mask.to(accelerator.device)

        optimizer.zero_grad(set_to_none=True)

        with accelerator.autocast():
            mels_out, mels_postnet_out, stop_preds, _ = model(
                texts, text_lens.to("cpu"), mels, encoder_mask, decoder_mask
            )

        loss, mel_loss, refined_mel_loss, stop_loss = compute_losses(
            mels_out, mels_postnet_out, stop_preds, mels, stops
        )

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if using_scheduler:
            scheduler.step()

        loss_value = torch.mean(accelerator.gather_for_metrics(loss.detach())).item()
        mel_loss_value = torch.mean(accelerator.gather_for_metrics(mel_loss.detach())).item()
        refined_mel_loss_value = torch.mean(accelerator.gather_for_metrics(refined_mel_loss.detach())).item()
        stop_loss_value = torch.mean(accelerator.gather_for_metrics(stop_loss.detach())).item()

        if completed_steps % args.console_out_iters == 0:
            accelerator.print(
                "Completed Steps {}/{} | Loss {:.4f} | Mel Loss {:.4f} | RMel Loss {:.4f} | Stop Loss {:.4f}".format(
                    completed_steps,
                    args.training_steps,
                    loss_value,
                    mel_loss_value,
                    refined_mel_loss_value,
                    stop_loss_value,
                )
            )

        if accelerator.is_main_process:
            append_csv_row(
                train_log_path,
                train_log_fields,
                {
                    "step": completed_steps,
                    "dataset_pass": dataset_pass,
                    "loss": loss_value,
                    "mel_loss": mel_loss_value,
                    "refined_mel_loss": refined_mel_loss_value,
                    "stop_loss": stop_loss_value,
                    "learning_rate": current_lr(),
                },
            )

        if completed_steps % args.wandb_log_iters == 0 and args.log_wandb:
            accelerator.log(
                {
                    "mel_loss": mel_loss_value,
                    "refined_mel_loss": refined_mel_loss_value,
                    "stop_loss": stop_loss_value,
                    "total_loss": loss_value,
                    "learning_rate": current_lr(),
                },
                step=completed_steps,
            )

        completed_steps += 1

        if completed_steps % args.eval_steps == 0 or completed_steps == args.training_steps:
            evaluate(completed_steps)

        if completed_steps % args.checkpoint_steps == 0 or completed_steps == args.training_steps:
            accelerator.print("Saving Checkpoint!")
            path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_step_{completed_steps}")
            accelerator.save_state(output_dir=path_to_checkpoint, safe_serialization=False)

    completed_passes += 1
    accelerator.print(
        f"Completed Dataset Pass Estimate: {completed_steps / steps_per_pass:.2f} | Learning Rate: {current_lr(): .3e}"
    )

accelerator.save_state(os.path.join(path_to_experiment, "final_checkpoint"), safe_serialization=False)
accelerator.end_training()
