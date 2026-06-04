import os
import argparse
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import trange

class MaskedCNN(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedCNN, self).__init__(*args, **kwargs)
        out_channels, in_channels, height, width = self.weight.shape
        self.register_buffer("mask", torch.ones_like(self.weight))
        
        if mask_type == "A":
            self.mask[:, :, height//2:, width//2:] = 0
            self.mask[:,:,height//2+1:,:] = 0
        elif mask_type == "B":
            self.mask[:, :, height//2:, width//2+1:] = 0
            self.mask[:,:,height//2+1:,:] = 0
        
    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedCNN, self).forward(x)

class PixelCNN(nn.Module):
    def __init__(self, 
                 input_channels=1, 
                 num_layers=8, 
                 kernel_size=7, 
                 channels=64,
                 bit_depth=8):
        super(PixelCNN, self).__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.channels = channels
        self.num_outputs = 2**bit_depth

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(
                    MaskedCNN(
                        mask_type="A",
                        in_channels=input_channels, 
                        out_channels=channels, 
                        kernel_size=kernel_size, 
                        stride=1, 
                        padding="same"
                    )
                )
            
            else:

                layers.append(
                    MaskedCNN(
                        mask_type="B",
                        in_channels=channels, 
                        out_channels=channels, 
                        kernel_size=kernel_size, 
                        stride=1, 
                        padding="same"
                    )
                )

            layers.extend(
                [nn.BatchNorm2d(channels), nn.ReLU()]
            )

        self.layers = nn.ModuleList(layers)

        self.out_proj = nn.Conv2d(in_channels=channels, 
                                  out_channels=input_channels * self.num_outputs, 
                                  kernel_size=1)
    
    def forward(self, x):

        batch, channels, height, width = x.shape

        for layer in self.layers:
            x = layer(x)
        
        output = self.out_proj(x)

        output = output.reshape(batch, channels, self.num_outputs, height, width)
        output = output.permute(0,2,1,3,4)
        
        return output

def generate_samples(model, num_samples=16, image_size=32, num_channels=3, device='cpu'):
    model.eval()
    
    samples = torch.zeros(num_samples, num_channels, image_size, image_size, device=device, dtype=torch.long)
    
    with torch.no_grad():
        for i in range(image_size):
            for j in range(image_size):
                # Get logits: [B, num_outputs, in_channels, H, W]
                logits = model(samples.float() / 255.0)

                for ch in range(num_channels):
                    logits_ch = logits[:, :, ch, i, j]  # [B, num_outputs] for this pixel/channel
                    probs = torch.softmax(logits_ch, dim=1)
                    samples[:, ch, i, j] = torch.multinomial(probs, 1).squeeze(-1)

    model.train()
    return samples