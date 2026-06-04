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

from pixel_cnn_model import PixelCNN, generate_samples

# --- Argument Parser ---
def get_args():
    parser = argparse.ArgumentParser(description="Train PixelCNN on CIFAR-10")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--data_dir", type=str, default="data/", help="Path to dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="work_dir/checkpoints", help="Where to save checkpoints")
    parser.add_argument("--gens_dir", type=str, default="work_dir/gens", help="Where to save generated samples")
    parser.add_argument("--log_csv", type=str, default=None, help="Optional CSV file for epoch metrics")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--sample_every", type=int, default=1, help="Generate samples every N epochs")
    parser.add_argument("--checkpoint_every", type=int, default=1, help="Save checkpoints every N epochs")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional batch cap for smoke tests")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 mixed precision training")
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.gens_dir, exist_ok=True)
    if args.log_csv is not None:
        os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
        with open(args.log_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "avg_loss", "num_batches", "lr"])

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
        pin_memory=args.device.startswith("cuda"),
    )

    # Model, optimizer
    model = PixelCNN(input_channels=input_channels, bit_depth=8).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(args)
    print(f"Dataset size: {len(train_dataset)} | Batches per epoch: {len(train_loader)}")

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")

        for batch_idx, (data, _) in enumerate(pbar):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            data = data.to(args.device)
            optimizer.zero_grad()

            with torch.amp.autocast(args.device, dtype=torch.bfloat16, enabled=args.bf16):
                logits = model(data)
                targets = (data * 255).to(torch.long)
                loss = F.cross_entropy(logits, targets)
                
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}")
        if args.log_csv is not None:
            with open(args.log_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, avg_loss, num_batches, optimizer.param_groups[0]["lr"]])

        # Generate samples
        if args.sample_every > 0 and ((epoch + 1) % args.sample_every == 0 or epoch + 1 == args.epochs):
            samples = generate_samples(model, num_samples=16, image_size=image_size, num_channels=input_channels, device=args.device)
            
            plt.figure(figsize=(8, 8))
            samples_float = samples.float() / 255.0
            grid = vutils.make_grid(samples_float, nrow=4, normalize=True)
            plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
            plt.title("Generated Samples")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(args.gens_dir, f"epoch_{epoch+1}.png"))
            plt.close()

        # Save checkpoint
        if args.checkpoint_every > 0 and ((epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.epochs):
            checkpoint_path = os.path.join(args.checkpoint_dir, f"pixelcnn_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main()
