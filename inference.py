import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

# ==========================================
# 1. Model Definition (Must match train_codec.py)
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
        self.encoder = nn.Sequential(
            CausalConv1d(1, channels, 7, stride=1),
            ResBlock(channels),
            CausalConv1d(channels, channels*2, 4, stride=2), 
            ResBlock(channels*2),
            CausalConv1d(channels*2, channels*4, 4, stride=2), 
            ResBlock(channels*4),
            CausalConv1d(channels*4, channels*8, 4, stride=2), 
        )
        self.bottleneck = nn.Conv1d(channels*8, channels*8, 3, padding=1)
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
        if out.shape[-1] != x.shape[-1]:
            out = out[..., :x.shape[-1]]
        return out

# ==========================================
# 2. Inference Logic (Chunked)
# ==========================================
def process_chunked(model, audio, chunk_sec=5.0, overlap_sec=0.5, sr=24000, device='cuda'):
    """
    Runs inference using Overlap-Add to handle long files and memory limits.
    """
    chunk_samples = int(chunk_sec * sr)
    overlap_samples = int(overlap_sec * sr)
    stride = chunk_samples - overlap_samples
    
    length = audio.shape[-1]
    
    # Output buffer
    output = torch.zeros_like(audio)
    # Weight buffer (to normalize the overlap additions)
    weights = torch.zeros_like(audio)
    
    # Create a linear fade window for the overlap regions
    # (Not strictly necessary for causal models if state is managed, 
    # but essential for stateless chunking to avoid clicks)
    # Ideally we use a window that sums to 1.
    
    current_idx = 0
    
    print(f"Processing {length/sr:.2f}s audio in {chunk_sec}s chunks...")
    
    while current_idx < length:
        # Determine End
        end_idx = min(current_idx + chunk_samples, length)
        
        # Extract Chunk
        chunk = audio[:, :, current_idx:end_idx].to(device)
        chunk_len = chunk.shape[-1]
        
        # Pad if last chunk is too small (optional, but helps Conv dimension math)
        pad_amt = 0
        if chunk_len < 8: # Minimum receptive field padding
             pad_amt = 8 - chunk_len
             chunk = F.pad(chunk, (0, pad_amt))
        
        # Inference
        with torch.no_grad():
            recon = model(chunk)
            
        # Remove Padding
        if pad_amt > 0:
            recon = recon[..., :-pad_amt]
        
        # Add to Output
        output[:, :, current_idx:end_idx] += recon.cpu()
        weights[:, :, current_idx:end_idx] += 1.0
        
        # Step Forward
        current_idx += stride
        
        # Progress Bar-ish
        if current_idx % (stride * 5) == 0:
            print(f"  -> {(current_idx/length)*100:.1f}%")
            
    # Normalize by weights (Averaging the overlaps)
    # Avoid div by zero
    weights[weights == 0] = 1.0
    output = output / weights
    
    return output

def run_inference(model_path, input_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Model from {model_path} on {device}...")
    
    # Load Model
    model = SimpleCodec().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load Audio
    print(f"Loading {input_path}...")
    audio, sr = torchaudio.load(input_path)
    
    # Convert to Mono
    if audio.shape[0] > 1:
        print("Stereo detected. Converting to Mono for processing...")
        audio = audio.mean(dim=0, keepdim=True)
        
    # Resample
    if sr != 24000:
        print(f"Resampling {sr} -> 24000...")
        audio = torchaudio.transforms.Resample(sr, 24000)(audio)
        
    # Normalize Input
    gain = audio.abs().max()
    if gain > 0:
        audio = audio / gain
        
    # Process
    # We create a batch dim [1, 1, T]
    audio_batch = audio.unsqueeze(0)
    
    recon_batch = process_chunked(model, audio_batch, chunk_sec=5.0, overlap_sec=0.1, device=device)
    
    recon = recon_batch.squeeze(0)
    
    # Renormalize Output (Safe Guard)
    recon = recon / (recon.abs().max() + 1e-6)
    
    print(f"Saving to {output_path}...")
    torchaudio.save(output_path, recon, 24000)
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inference_codec.py <model_checkpoint.pth> <input.wav> [output.wav]")
        sys.exit(1)
        
    model_p = sys.argv[1]
    input_p = sys.argv[2]
    
    if len(sys.argv) >= 4:
        output_p = sys.argv[3]
    else:
        base, _ = os.path.splitext(input_p)
        output_p = f"{base}_recon.wav"
        
    run_inference(model_p, input_p, output_p)