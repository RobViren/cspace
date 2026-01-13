import sys
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from cspace_model import CSpace
import config

def load_and_prep(path, target_sr=24000):
    audio, sr = torchaudio.load(path)
    # Mono
    if audio.shape[0] > 1: audio = audio.mean(dim=0, keepdim=True)
    # Resample
    if sr != target_sr: audio = torchaudio.transforms.Resample(sr, target_sr)(audio)
    return audio

def calculate_si_snr(preds, target):
    """Scale-Invariant Signal-to-Noise Ratio"""
    # Zero-mean
    target = target - torch.mean(target, dim=-1, keepdim=True)
    preds = preds - torch.mean(preds, dim=-1, keepdim=True)
    
    # Scaling factor (alpha)
    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) / 
             (torch.sum(target ** 2, dim=-1, keepdim=True) + 1e-8))
    
    target_scaled = alpha * target
    noise = preds - target_scaled
    
    val = 10 * torch.log10(torch.sum(target_scaled ** 2, dim=-1) / 
                           (torch.sum(noise ** 2, dim=-1) + 1e-8))
    return val.item()

def run_metrics(ref_path, deg_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Comparing:\n  Ref: {ref_path}\n  Deg: {deg_path}")
    
    # Load
    ref = load_and_prep(ref_path).to(device)
    deg = load_and_prep(deg_path).to(device)
    
    # Crop to minimum length to ensure alignment
    min_len = min(ref.shape[-1], deg.shape[-1])
    ref = ref[..., :min_len]
    deg = deg[..., :min_len]
    
    # --- 1. Time Domain MSE ---
    # We normalize energy first to make MSE meaningful (Volume independent)
    ref_norm = ref / ref.abs().max()
    deg_norm = deg / deg.abs().max()
    
    mse = F.mse_loss(deg_norm, ref_norm).item()
    
    # --- 2. SI-SNR ---
    # Industry standard (Decibels)
    snr = calculate_si_snr(deg, ref)
    
    # --- 3. C-Space Distance (The Geometric Error) ---
    cspace = CSpace(res_size=64).to(device)
    with torch.no_grad():
        ref_c = cspace(ref)
        deg_c = cspace(deg)
        # L1 Error in C-Space
        c_dist = F.l1_loss(deg_c, ref_c).item()

    print("-" * 40)
    print(f"Time Domain MSE:   {mse:.6f} (Lower is Better)")
    print(f"C-Space Error:     {c_dist:.6f} (Lower is Better)")
    print(f"SI-SNR (dB):       {snr:.2f}   (Higher is Better)")
    print("-" * 40)
    
    if snr > 15:
        print("Verdict: High Fidelity")
    elif snr > 5:
        print("Verdict: Good/Intelligible")
    else:
        print("Verdict: Poor/Artifacts")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python measure_fidelity.py <original.wav> <reconstructed.wav>")
        sys.exit(1)
    run_metrics(sys.argv[1], sys.argv[2])