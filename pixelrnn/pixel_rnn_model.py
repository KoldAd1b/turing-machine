import torch
import torch.nn as nn

class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, mask_type, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
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
        return super(MaskedConv2d, self).forward(x)

class RowLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, mask_type='B', image_size=32, train_init_states=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.is_proj = MaskedConv2d(in_channels, 4 * hidden_channels, kernel_size=(1, kernel_size), padding="same", mask_type=mask_type)
        self.ss_proj = MaskedConv2d(hidden_channels, 4 * hidden_channels, kernel_size=(1, kernel_size), padding="same", mask_type=mask_type)

        self.h0 = nn.Parameter(torch.zeros(1, hidden_channels, 1, image_size), requires_grad=train_init_states)
        self.c0 = nn.Parameter(torch.zeros(1, hidden_channels, 1, image_size), requires_grad=train_init_states)
        
    def forward(self, x):
        
        b, c, height, width = x.shape

        x = self.is_proj(x)

        ### Create copies of Hidden and Cell State ###
        c_prev = self.c0.expand(b, -1, -1, -1).contiguous()  # [B, H*hidden_dim]
        h_prev = self.h0.expand(b, -1, -1, -1).contiguous()  # [B, H*hidden_dim]

        ### To store final outputs ###
        hs = []
        for row in range(height):

            ### Grab Row ###
            row_x = x[:, :, row:row + 1, :]

            ### Project Previous h ###
            proj_h = self.ss_proj(h_prev)

            ### Follow Formulation (3) in paper ###
            gates_raw = torch.sigmoid(row_x + proj_h)
            o, f, i, g = torch.split(gates_raw, self.hidden_channels, dim=1)
            c = f * c_prev + i * g
            h = o * torch.tanh(c)

            ### Store output ###
            hs.append(h)

            ### Update Hidden and Cell State ###
            h_prev = h
            c_prev = c

        ### Concat together everything ###
        hidden = torch.cat(hs, dim=2)  # [b, h, height, width]
        
        return hidden

class ResidualRowLSTM(nn.Module):
    def __init__(self, 
                 hidden_channels, 
                 kernel_size=3, 
                 image_size=28):
        
        super().__init__()
        self.hidden_channels = hidden_channels
        self.row_lstm = RowLSTM(2 * hidden_channels, hidden_channels, kernel_size, image_size=image_size)
        self.out_conv = MaskedConv2d(hidden_channels, 2 * hidden_channels, kernel_size=(1, 1), padding="same", mask_type='B')

    def forward(self, x):
        hidden = self.row_lstm(x)
        out = self.out_conv(hidden) + x
        return out

class PixelRNN(nn.Module):
    def __init__(self, num_layers=12, hidden_channels=64, input_channels=3, bit_depth=8, image_size=32):
        super().__init__()

        self.num_outputs = 2 ** bit_depth
        self.input_channels = input_channels
        
        self.input_conv = MaskedConv2d(input_channels, 2 * hidden_channels, kernel_size=(7, 7), padding="same", mask_type='A')
        self.layers = nn.ModuleList([ResidualRowLSTM(hidden_channels, image_size=image_size) for _ in range(num_layers)])
        self.out1 = MaskedConv2d(2 * hidden_channels, 2 * hidden_channels, kernel_size=(1, 1), padding="same", mask_type='B')
        self.out2 = MaskedConv2d(2 * hidden_channels, 2 * hidden_channels, kernel_size=(1, 1), padding="same", mask_type='B')
        self.final_conv = MaskedConv2d(2 * hidden_channels, input_channels * self.num_outputs, kernel_size=(1, 1), padding="same", mask_type='B')

    def forward(self, x):

        batch, channels, height, width = x.shape

        ### Projection with Maske type A ###
        features = self.input_conv(x)

        ### Recurrent (conv) Layers ###
        for layer in self.layers:
            features = layer(features)

        ### Final Output Projections (Mask B) ###
        out = torch.relu(features)
        out = self.out1(out)
        out = torch.relu(out)
        out = self.out2(out)

        ### Project to Output Space (B x in_channels * num_outputs, H, W) ###
        logits = self.final_conv(out)  
        logits = logits.reshape(batch, self.input_channels, self.num_outputs, height, width)

        ### Permute Output (B x num_outputs x in_channels, H, W) ###
        logits = logits.permute(0,2,1,3,4)
        
        return logits

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

if __name__ == "__main__":
    model = PixelRNN()
    print(model)