import os
import random
import argparse
import csv
from PIL import Image
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator

### Load in My UNET and DataLoader ###
from unet_train_parts import ADE20KDataset, UNET

### Define Simple Training Logger ###
class LocalLogger:
    def __init__(self, 
                 path_to_log_folder, 
                 filename="train_log.pkl"):
        
        self.path_to_log_folder = path_to_log_folder
        self.path_to_file = os.path.join(path_to_log_folder, filename)

        self.log_exists = os.path.isfile(self.path_to_file)

        if self.log_exists:
            with open(self.path_to_file, "rb") as f:
                self.logger = pickle.load(f)
            
        else:
            self.logger = {"epoch": [], 
                           "train_loss": [], 
                           "train_acc": [], 
                           "test_loss": [], 
                           "test_acc": []}
            
    def log(self, epoch, train_loss, train_acc, test_loss, test_acc):
        self.logger["epoch"].append(epoch)
        self.logger["train_loss"].append(train_loss)
        self.logger["train_acc"].append(train_acc)
        self.logger["test_loss"].append(test_loss)
        self.logger["test_acc"].append(test_acc)

        with open(self.path_to_file, "wb") as f:
            pickle.dump(self.logger, f)


def append_csv_row(path_to_file, fieldnames, row):
    write_header = not os.path.isfile(path_to_file)
    with open(path_to_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


### Write Training Function ###
def train(batch_size=64, 
          gradient_accumulation_steps=2,
          learning_rate=0.001, 
          num_epochs=150,
          image_size=256,
          path_to_data="../../data/ADE20K",
          experiment_name="unet_w_skip_ade20k",
          skip_connection=True,
          num_workers=16,
          max_train_steps=None,
          max_val_steps=None,
          log_every_n_steps=10,
          working_directory=None):

    ### Define Accelerator ###
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

    ## Create Working Directory ###
    if working_directory is None:
        working_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work_dir")
    path_to_experiment = os.path.join(working_directory, experiment_name)
    os.makedirs(path_to_experiment, exist_ok=True)

    ### Instantiate Logger ###
    logger = LocalLogger(path_to_experiment, f"{experiment_name}_log.pkl")
    step_log_path = os.path.join(path_to_experiment, "train_steps.csv")
    epoch_log_path = os.path.join(path_to_experiment, "epoch_metrics.csv")

    ### Load Dataset ###
    micro_batchsize = batch_size // gradient_accumulation_steps
    train_data = ADE20KDataset(path_to_data, train=True, image_size=image_size)
    test_data = ADE20KDataset(path_to_data, train=False, image_size=image_size)
    train_dataloader = DataLoader(train_data, batch_size=micro_batchsize, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=micro_batchsize, shuffle=False, num_workers=num_workers, pin_memory=True)

    ### Define Loss Function (ignore index -1 as its unlabeled background) ###
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    ### Load Model ###
    model = UNET(in_channels=3, 
                 num_classes=150, 
                 start_dim=64, 
                 dim_mults=(1,2,4,8),
                 skip_connection=skip_connection)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    accelerator.print("Number of Parameters:", params)
    
    ### Load Optimizer ###
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    ### Load Scheduler ###
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=500, 
                                                num_training_steps=(len(train_dataloader) * num_epochs))
    
    ### Prepare Everything ###
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )
    
    ### Train Model ###
    best_test_loss = np.inf
    global_step = 0

    for epoch in range(1,num_epochs+1):

        accelerator.print(f"Training Epoch [{epoch}/{num_epochs}]")
        
        train_loss, test_loss = [], []
        train_acc, test_acc = [], []

        ### Train Loop ###
        accumulated_loss = 0 
        accumulated_accuracy = 0
        total_train_steps = len(train_dataloader)//gradient_accumulation_steps
        if max_train_steps is not None:
            total_train_steps = min(total_train_steps, max_train_steps)
        progress_bar = tqdm(range(total_train_steps), disable = not accelerator.is_main_process)
        
        model.train()
        completed_train_steps = 0
        for images, targets in train_dataloader:
            
            with accelerator.accumulate(model):
                
                ### Pass Through Model ###
                pred = model(images)

                ### Compute and Store Loss (Scaled by Grad Accumulations) ##
                loss = loss_fn(pred, targets)
                accumulated_loss += loss / gradient_accumulation_steps

                ### Compute and Store Accuracy ###
                predicted = pred.argmax(axis=1)
                accuracy = (predicted == targets).sum() / torch.numel(predicted)
                accumulated_accuracy += accuracy / gradient_accumulation_steps

                ### Compute Gradients ###
                accelerator.backward(loss)

                ### Gradient Clipping and Logging ###
                if accelerator.sync_gradients:
                    
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                    ### Gather Metrics Across GPUs ###
                    loss_gathered = accelerator.gather_for_metrics(accumulated_loss)
                    accuracy_gathered = accelerator.gather_for_metrics(accumulated_accuracy)

                    ### Store Current Iteration Loss and Accuracy ###
                    step_loss = torch.mean(loss_gathered).item()
                    step_acc = torch.mean(accuracy_gathered).item()
                    train_loss.append(step_loss)
                    train_acc.append(step_acc)
                    global_step += 1

                    if accelerator.is_main_process and (global_step == 1 or global_step % log_every_n_steps == 0):
                        append_csv_row(
                            step_log_path,
                            ["global_step", "epoch", "train_loss", "train_acc"],
                            {"global_step": global_step, "epoch": epoch, "train_loss": step_loss, "train_acc": step_acc},
                        )
                        progress_bar.set_postfix(loss=f"{step_loss:.4f}", acc=f"{step_acc:.4f}")
                        accelerator.print(f"step={global_step} epoch={epoch} train_loss={step_loss:.4f} train_acc={step_acc:.4f}")

                    ### Reset Accumulated for Next Accumulation ###
                    accumulated_loss, accumulated_accuracy = 0, 0

                    ### Iterate Progress Bar ###
                    progress_bar.update(1)
                    completed_train_steps += 1

                ### Update Model ###
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if max_train_steps is not None and completed_train_steps >= max_train_steps:
                    break

            if max_train_steps is not None and completed_train_steps >= max_train_steps:
                break
        
        ### Testing Loop ###
        model.eval()
        for val_step, (images, targets) in enumerate(test_dataloader):

            if max_val_steps is not None and val_step >= max_val_steps:
                break

            with torch.no_grad():
                pred = model(images)

            ### Compute Loss ###
            loss = loss_fn(pred, targets)

            ### Compute Accuracy ###
            predicted = pred.argmax(axis=1)
            accuracy = (predicted == targets).sum() / torch.numel(predicted)

            ### Gather Losses and Accuracy ###
            loss_gathered = accelerator.gather_for_metrics(loss)
            accuracy_gathered = accelerator.gather_for_metrics(accuracy)

            ### Store Current Iteration Error ###
            test_loss.append(torch.mean(loss_gathered).item())
            test_acc.append(torch.mean(accuracy_gathered).item())

        ### Average Loss and Acc for Epoch ###
        epoch_train_loss = np.mean(train_loss)
        epoch_test_loss = np.mean(test_loss)
        epoch_train_acc = np.mean(train_acc)
        epoch_test_acc = np.mean(test_acc)

        accelerator.print(f"Training Accuracy: {epoch_train_acc}, Training Loss: {epoch_train_loss}")
        accelerator.print(f"Testing Accuracy: {epoch_test_acc}, Testing Loss: {epoch_test_loss}")

        ### Log Training ###
        if accelerator.is_main_process:
            logger.log(epoch=epoch,
                       train_loss=epoch_train_loss,
                       train_acc=epoch_train_acc,
                       test_loss=epoch_test_loss,
                       test_acc=epoch_test_acc)
            append_csv_row(
                epoch_log_path,
                ["epoch", "train_loss", "train_acc", "test_loss", "test_acc"],
                {
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "train_acc": epoch_train_acc,
                    "test_loss": epoch_test_loss,
                    "test_acc": epoch_test_acc,
                },
            )
        
        ### Save Model ###
        if epoch_test_loss < best_test_loss:
            accelerator.print("---SAVING---")

            best_test_loss = epoch_test_loss
            accelerator.save_model(model, os.path.join(path_to_experiment, "best_checkpoint"), safe_serialization=False)

        accelerator.save_model(model, os.path.join(path_to_experiment, "last_checkpoint"), safe_serialization=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet on ADE20K semantic segmentation")
    parser.add_argument("--path_to_data", type=str, default="../../data/ADE20K")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--experiment_name", type=str, default="UNET_wo_skip_ADE20K_test")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_val_steps", type=int, default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--working_directory", type=str, default=None)
    parser.add_argument("--no_skip_connection", action="store_true")
    args = parser.parse_args()

    train(batch_size=args.batch_size,
          gradient_accumulation_steps=args.gradient_accumulation_steps,
          learning_rate=args.learning_rate,
          num_epochs=args.num_epochs,
          image_size=args.image_size,
          path_to_data=args.path_to_data,
          experiment_name=args.experiment_name,
          skip_connection=not args.no_skip_connection,
          num_workers=args.num_workers,
          max_train_steps=args.max_train_steps,
          max_val_steps=args.max_val_steps,
          log_every_n_steps=args.log_every_n_steps,
          working_directory=args.working_directory)
                




    

 
