import sys
import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import librosa
import matplotlib.pyplot as plt
from cspace_model import CSpace
import config

# --- Loss Definitions ---

class MelSpecLoss(nn.Module):
    def __init__(self, sr=24000):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80, normalized=True
        )
    def forward(self, pred, target):
        # Log-Mel L1 Loss
        p_m = self.mel(pred).log1p()
        t_m = self.mel(target).log1p()
        return torch.nn.functional.l1_loss(p_m, t_m)

class MultiScaleSpecLoss(nn.Module):
    def __init__(self, sr=24000):
        super().__init__()
        self.scales = [512, 1024, 2048]
        self.mels = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=s, hop_length=s//4, n_mels=64, normalized=True
            ) for s in self.scales
        ])
    def forward(self, pred, target):
        loss = 0.0
        for mel in self.mels:
            mel = mel.to(pred.device)
            loss += torch.nn.functional.l1_loss(mel(pred).log1p(), mel(target).log1p())
        return loss

class HolographicLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.cspace = CSpace(res_size=64,min_freq=10,max_freq=12000).to(device)
    
    def forward(self, pred, target):
        # L1 Loss on C-Space States
        # States are [RealF, ImagF, RealB, ImagB]
        p_s = self.cspace(pred)
        t_s = self.cspace(target)
        return torch.nn.functional.l1_loss(p_s, t_s)

# --- Comparison Engine ---

def compare_landscapes(wav_path, center_idx, range_samples=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Audio
    audio, _ = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
    audio = audio / np.max(np.abs(audio))
    audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Define Losses
    losses = {
        "MelSpectrogram": MelSpecLoss(config.SAMPLE_RATE).to(device),
        "Multi-Scale Spectral": MultiScaleSpecLoss(config.SAMPLE_RATE).to(device),
        "C-Space Holographic": HolographicLoss(device).to(device)
    }
    
    # Define Window Size for Comparison (e.g., 2048 samples)
    win_size = 2048
    half_win = win_size // 2
    
    # Target Window (Fixed)
    target_s = center_idx - half_win
    target_e = center_idx + half_win
    target_chunk = audio[:, target_s:target_e]
    
    # Arrays to store results
    results = {k: [] for k in losses.keys()}
    offsets = np.arange(-range_samples, range_samples + 1)
    
    print(f"Sweeping offsets {offsets[0]} to {offsets[-1]} around sample {center_idx}...")
    
    for offset in offsets:
        # Shifted Pred Window
        s = target_s + offset
        e = target_e + offset
        pred_chunk = audio[:, s:e]
        
        with torch.no_grad():
            for name, criterion in losses.items():
                loss_val = criterion(pred_chunk, target_chunk).item()
                results[name].append(loss_val)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    for name, vals in results.items():
        vals = np.array(vals)
        # Min-Max Normalization to compare Topology
        norm_vals = (vals - vals.min()) / (vals.max() - vals.min())
        
        # Style
        if "C-Space" in name:
            plt.plot(offsets, norm_vals, label=name, linewidth=2.5, color='blue')
        elif "Multi" in name:
            plt.plot(offsets, norm_vals, label=name, linestyle='--', color='orange')
        else:
            plt.plot(offsets, norm_vals, label=name, linestyle=':', color='gray')
            
    plt.title(f"Loss Landscape Topology Comparison (Center: {center_idx})")
    plt.xlabel("Phase Offset (Samples)")
    plt.ylabel("Normalized Loss (0-1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(0, color='red', alpha=0.2)
    
    if not os.path.exists(config.RESULTS_DIR): os.makedirs(config.RESULTS_DIR)
    out_path = os.path.join(config.RESULTS_DIR, "loss_topology_comparison.png")
    plt.savefig(out_path)
    print(f"Saved comparison to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3: sys.exit(1)
    compare_landscapes(sys.argv[1], int(sys.argv[2]))