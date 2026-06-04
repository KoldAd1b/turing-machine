import argparse
import csv
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator

from pixel_rnn_model import PixelRNN, generate_samples

# --- Argument Parser ---
def get_args():
    parser = argparse.ArgumentParser(description="Train PixelRNN on CIFAR-10")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--data_dir", type=str, default="data/", help="Path to dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="work_dir/checkpoints", help="Where to save checkpoints")
    parser.add_argument("--gens_dir", type=str, default="work_dir/gens", help="Where to save generated samples")
    parser.add_argument("--log_csv", type=str, default=None, help="Optional CSV file for epoch metrics")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--sample_every", type=int, default=5, help="Generate samples every N epochs")
    parser.add_argument("--num_samples", type=int, default=16, help="Generated samples per sample grid")
    parser.add_argument("--checkpoint_every", type=int, default=5, help="Save checkpoints every N epochs")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional batch cap for smoke tests")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 mixed precision training")
    return parser.parse_args()


def main():
    args = get_args()
    accelerator = Accelerator(project_dir=os.path.dirname(args.checkpoint_dir))
    if accelerator.is_main_process:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.gens_dir, exist_ok=True)
    if args.log_csv is not None and accelerator.is_main_process:
        os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
        with open(args.log_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "avg_loss", "num_batches_per_process", "num_samples", "lr"])

    # Data loading
    transform = transforms.Compose([transforms.ToTensor()])

    if args.dataset == "mnist":
        train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
        input_channels = 1
        image_size = 28
    elif args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
        input_channels = 3
        image_size = 32
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Model, optimizer
    model = PixelRNN(input_channels=input_channels, bit_depth=8, image_size=image_size)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    accelerator.print(args)
    accelerator.print(
        f"Dataset size: {len(train_dataset)} | "
        f"Batches per process per epoch: {len(train_loader)} | "
        f"Processes: {accelerator.num_processes}"
    )

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = torch.tensor(0.0, device=accelerator.device)
        total_count = torch.tensor(0.0, device=accelerator.device)
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch", disable=not accelerator.is_main_process)

        for batch_idx, (data, _) in enumerate(pbar):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            optimizer.zero_grad()

            with accelerator.autocast():
                logits = model(data)
                targets = (data * 255).to(torch.long)
                loss = F.cross_entropy(logits, targets)
                
            accelerator.backward(loss)
            optimizer.step()

            batch_size = data.shape[0]
            total_loss += loss.detach() * batch_size
            total_count += batch_size
            num_batches += 1
            if accelerator.is_main_process:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_stats = torch.stack([total_loss, total_count])
        epoch_stats = accelerator.gather(epoch_stats).reshape(-1, 2).sum(dim=0)
        avg_loss = (epoch_stats[0] / epoch_stats[1]).item()
        samples_seen = int(epoch_stats[1].item())
        accelerator.print(f"Epoch {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}")
        if args.log_csv is not None and accelerator.is_main_process:
            with open(args.log_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, avg_loss, num_batches, samples_seen, optimizer.param_groups[0]["lr"]])

        # Generate samples
        if accelerator.is_main_process and args.sample_every > 0 and ((epoch + 1) % args.sample_every == 0 or epoch + 1 == args.epochs):
            unwrapped_model = accelerator.unwrap_model(model)
            samples = generate_samples(unwrapped_model, num_samples=args.num_samples, image_size=image_size, num_channels=input_channels, device=accelerator.device)
            
            plt.figure(figsize=(8, 8))
            samples_float = samples.float() / 255.0
            grid = vutils.make_grid(samples_float, nrow=max(1, int(args.num_samples ** 0.5)), normalize=True)
            plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
            plt.title("Generated Samples")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(args.gens_dir, f"epoch_{epoch+1}.png"))
            plt.close()

        # Save checkpoint
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and args.checkpoint_every > 0 and ((epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.epochs):
            checkpoint_path = os.path.join(args.checkpoint_dir, f"pixelrnn_epoch_{epoch+1}.pth")
            torch.save(accelerator.unwrap_model(model).state_dict(), checkpoint_path)
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
