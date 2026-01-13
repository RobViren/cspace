import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import numpy as np
from cspace_loss import CSpaceLoss
import config

# ==========================================
# 1. Model Architecture (SimpleCodec)
# ==========================================
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ELU(),
            CausalConv1d(channels, channels, 3, dilation=dilation),
            nn.ELU(),
            CausalConv1d(channels, channels, 1)
        )
    def forward(self, x):
        return x + self.block(x)

class SimpleCodec(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        
        # Encoder: 1 -> 32 -> 64 -> 128 (Downsample 8x total)
        self.encoder = nn.Sequential(
            CausalConv1d(1, channels, 7, stride=1),
            ResBlock(channels),
            CausalConv1d(channels, channels*2, 4, stride=2), 
            ResBlock(channels*2),
            CausalConv1d(channels*2, channels*4, 4, stride=2), 
            ResBlock(channels*4),
            CausalConv1d(channels*4, channels*8, 4, stride=2), 
        )
        
        # Continuous Bottleneck
        self.bottleneck = nn.Conv1d(channels*8, channels*8, 3, padding=1)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels*8, channels*4, 4, stride=2),
            ResBlock(channels*4),
            nn.ConvTranspose1d(channels*4, channels*2, 4, stride=2),
            ResBlock(channels*2),
            nn.ConvTranspose1d(channels*2, channels, 4, stride=2),
            ResBlock(channels),
            nn.Conv1d(channels, 1, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.bottleneck(z)
        out = self.decoder(z)
        # Fix slight shape mismatch from striding/padding
        if out.shape[-1] != x.shape[-1]:
            out = out[..., :x.shape[-1]]
        return out

# ==========================================
# 2. Data Loading (Streaming)
# ==========================================
class LibriStream(IterableDataset):
    def __init__(self, split="train.clean.100", segment_size=24000):
        print(f"Initializing LibriTTS Stream ({split})...")
        self.ds = load_dataset("mythicinfinity/libritts", "clean", split=split, streaming=True)
        self.segment_size = segment_size
        self.sr = 24000
        
    def __iter__(self):
        for item in self.ds:
            try:
                audio = item['audio']['array']
                orig_sr = item['audio']['sampling_rate']
                
                # Numpy -> Tensor
                audio = torch.from_numpy(audio).float()
                
                # Resample
                if orig_sr != self.sr:
                    resampler = torchaudio.transforms.Resample(orig_sr, self.sr)
                    audio = resampler(audio.unsqueeze(0)).squeeze(0)
                
                # Chunking
                if audio.shape[0] > self.segment_size:
                    # Yield 3 random chunks from this file to utilize the network load
                    for _ in range(3):
                        max_start = audio.shape[0] - self.segment_size
                        start = torch.randint(0, max_start, (1,)).item()
                        chunk = audio[start : start + self.segment_size]
                        
                        # Normalize chunk
                        m = chunk.abs().max()
                        if m > 0: chunk = chunk / m
                        
                        yield chunk.unsqueeze(0) # [1, T]
            except Exception as e:
                continue

# ==========================================
# 3. Validation Helper
# ==========================================
def load_validation_sample(path, device):
    print(f"Loading validation reference: {path}")
    audio, sr = torchaudio.load(path)
    # Mono
    if audio.shape[0] > 1: audio = audio.mean(dim=0, keepdim=True)
    # Resample
    if sr != 24000:
        audio = torchaudio.transforms.Resample(sr, 24000)(audio)
    # Crop to reasonable length (e.g. 10s max) to save inference time
    if audio.shape[-1] > 240000:
        audio = audio[..., :240000]
    # Normalize
    audio = audio / audio.abs().max()
    return audio.unsqueeze(0).to(device) # [1, 1, T]

# ==========================================
# 4. Training Loop
# ==========================================
def train(validation_path):
    # Config
    BATCH_SIZE = 8
    LR = 2e-4
    MAX_STEPS = 100000
    CHECKPOINT_INTERVAL = 500
    LOG_INTERVAL = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training Codec on {device} ---")
    
    # Init Data
    ds = LibriStream()
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0) # Streaming constraint
    
    # Init Model
    model = SimpleCodec().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # Init Loss (The C-Space Magic)
    criterion = CSpaceLoss(device=device, sr=24000, res_size=64).to(device)
    
    # Load Validation File
    val_audio = load_validation_sample(validation_path, device)
    
    # Save Dir
    os.makedirs("results/training", exist_ok=True)
    # Save original target for reference
    torchaudio.save("results/training/target_reference.wav", val_audio.squeeze().cpu(), 24000)
    
    step = 0
    iterator = iter(loader)
    
    print("Starting Training Loop...")
    
    while step < MAX_STEPS:
        try:
            x = next(iterator).to(device)
        except StopIteration:
            iterator = iter(loader)
            x = next(iterator).to(device)
            
        # Forward
        x_recon = model(x)
        
        # Loss
        loss = criterion(x_recon, x)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Logging
        if step % LOG_INTERVAL == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
            
        # Checkpoint
        if step % CHECKPOINT_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                # Reconstruct Validation File
                val_recon = model(val_audio)
                
                # Save
                path = f"results/training/step_{step}_recon.wav"
                # Normalize output to prevent clipping in wav file
                out_wav = val_recon.squeeze().cpu()
                out_wav = out_wav / (out_wav.abs().max() + 1e-6)
                
                torchaudio.save(path, out_wav.unsqueeze(0), 24000)
                
                # Save Model Weights
                torch.save(model.state_dict(), "results/training/codec_checkpoint.pth")
                
            print(f"--> Saved validation checkpoint to {path}")
            model.train()
            
        step += 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_codec.py <validation_wav_path>")
        sys.exit(1)
        
    train(sys.argv[1])