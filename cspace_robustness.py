import sys
import torch
import numpy as np
import librosa
from cspace_model import CSpace
import config

# Configuration Suite
# Name, Nodes, MinFreq, MaxFreq, Decay, HighFreqDecay
EXPERIMENTS = [
    ("Baseline",       64,   10,    12000, 0.999, 0.6),
    ("High Res",       256,  10,    12000, 0.999, 0.6),
    ("Tiny",           16,   10,    12000, 0.999, 0.6),
    ("Sub-Bass Only",  64,   10,    100,   0.999, 0.9), # Zoomed in ruler
    ("Treble Only",    64,   5000,  12000, 0.8,   0.4), # Zoomed in ruler
    ("Super Resonant", 64,   10,    12000, 0.9999,0.99),# Rings forever
]

def load_audio_torch(path):
    audio, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    max_val = np.max(np.abs(audio))
    if max_val > 0: audio = audio / max_val
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

def run_suite(wav_path, idx):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    audio = load_audio_torch(wav_path).to(device)
    
    print(f"\n--- C-Space Robustness Suite ---")
    print(f"Target Index: {idx} | Device: {device}")
    print("-" * 105)
    print(f"{'Experiment':<18} | {'Nodes':<5} | {'Range (Hz)':<12} | {'Decay':<6} | {'Match':<8} | {'Error':<5} | {'Sharpness':<10}")
    print("-" * 105)

    for name, nodes, f_min, f_max, d_low, d_high in EXPERIMENTS:
        # Init Model
        model = CSpace(res_size=nodes, min_freq=f_min, max_freq=f_max, 
                       decay_factor=d_low, high_freq_decay=d_high).to(device)
        
        with torch.no_grad():
            # Forward
            states = model(audio) # (1, Channels, Time)
            
            # Extract Query
            q_vec = states[0, :, idx]
            
            # Calculate Distance Matrix (L2 Norm over channels)
            # Target: (Channels, Time)
            # Query: (Channels, 1)
            diff = states[0] - q_vec.unsqueeze(1)
            dists = torch.linalg.vector_norm(diff, ord=2, dim=0).cpu().numpy()
            
            # Find Match
            match_idx = np.argmin(dists)
            error = match_idx - idx
            
            # Sharpness (L2 distance at 1 sample offset)
            sharp = dists[idx+1] if idx+1 < len(dists) else 0
            
            print(f"{name:<18} | {nodes:<5} | {f_min}-{f_max:<6} | {d_low:<6} | {match_idx:<8} | {error:<5} | {sharp:.5f}")

if __name__ == "__main__":
    if len(sys.argv) < 3: sys.exit(1)
    run_suite(sys.argv[1], int(sys.argv[2]))