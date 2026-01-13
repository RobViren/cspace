import sys
import os
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
from cspace_model import CSpace
import config

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_audio_torch(path):
    print(f"Loading {path}...")
    audio, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    max_val = np.max(np.abs(audio))
    if max_val > 0: audio = audio / max_val
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0) # (1, Time)

def plot_search_analysis(q_audio, t_audio, dist_total, dist_fwd, dist_bwd, q_idx, m_idx, fname):
    w = config.WINDOW_SAMPLES
    hw = w // 2
    
    # 1. Slice Waveforms for Overlay
    # Handle boundary conditions
    s_q, e_q = max(0, q_idx-hw), min(q_audio.shape[-1], q_idx+hw)
    s_m, e_m = max(0, m_idx-hw), min(t_audio.shape[-1], m_idx+hw)
    
    q_chunk = q_audio.squeeze().numpy()[s_q:e_q]
    m_chunk = t_audio.squeeze().numpy()[s_m:e_m]
    
    # 2. Slice Distances (Centered on Match)
    # We want to see the "Bowl" around the match index
    d_total = dist_total[s_m:e_m]
    d_fwd = dist_fwd[s_m:e_m]
    d_bwd = dist_bwd[s_m:e_m]
    
    # Create time axis relative to match
    t_axis = np.arange(len(m_chunk)) - (m_idx - s_m) # 0 is the match point

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
    
    # --- Plot 1: Waveform Alignment ---
    ax1.plot(t_axis[:len(q_chunk)], q_chunk, 'k', linewidth=1.5, label='Query')
    ax1.plot(t_axis, m_chunk, 'r--', linewidth=1.5, label='Match')
    ax1.set_title(f"Waveform Alignment (Offset: {m_idx - q_idx})")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: The "Cursed" Loss Landscape ---
    # Normalize for viz comparison so they fit on same Y-scale
    # (Optional, but helps see the shape intersection)
    
    ax2.plot(t_axis, d_fwd, color='orange', linestyle=':', label='Forward Loss (Past Context)')
    ax2.plot(t_axis, d_bwd, color='green', linestyle=':', label='Backward Loss (Future Context)')
    ax2.plot(t_axis, d_total, color='blue', linewidth=2, label='Total C-Space Distance')
    
    ax2.set_title("C-Space Loss Landscape (Phase Locking Basin)")
    ax2.set_ylabel("Euclidean Distance")
    ax2.set_xlabel("Offset from Match (Samples)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Mark the bottom of the bowl
    min_val = d_total[np.argmin(d_total)]
    ax2.scatter([0], [min_val], c='red', zorder=10)
    
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"Saved visualization to {fname}")

def run(q_path, t_path, q_idx):
    # Load
    q_audio = load_audio_torch(q_path)
    t_audio = load_audio_torch(t_path)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    # Note: Overshoot settings from config/model defaults
    model = CSpace(res_size=64).to(device)
    
    print("Processing C-Space...")
    with torch.no_grad():
        # Forward Pass
        # Shapes: (1, 256, Time)
        q_states = model(q_audio.to(device))
        t_states = model(t_audio.to(device))
        
        # Extract Query Vector at Index
        # Shape: (256)
        q_vec = q_states[0, :, q_idx]
        
        # --- Calculate Distances ---
        # We want to calculate the norm across the Channel dimension (dim=1)
        # Target: (1, 256, Time)
        # Query:  (256) -> broadcast to (1, 256, Time)
        
        diff = t_states - q_vec.unsqueeze(0).unsqueeze(2)
        
        # 1. Total Distance (L2 Norm of all 256 channels)
        # This is effectively sqrt(RealF^2 + ImagF^2 + RealB^2 + ImagB^2)
        dist_total = torch.linalg.vector_norm(diff, ord=2, dim=1).squeeze().cpu().numpy()
        
        # 2. Split Distances for Visualization
        # Channels 0-127 are Forward (Real/Imag), 128-255 are Backward
        mid = 128
        diff_fwd = diff[:, :mid, :]
        diff_bwd = diff[:, mid:, :]
        
        dist_fwd = torch.linalg.vector_norm(diff_fwd, ord=2, dim=1).squeeze().cpu().numpy()
        dist_bwd = torch.linalg.vector_norm(diff_bwd, ord=2, dim=1).squeeze().cpu().numpy()
        
        # --- Find Best Match ---
        match_idx = np.argmin(dist_total)
        print(f"Query Sample: {q_idx}")
        print(f"Best Match:   {match_idx}")
        print(f"Distance:     {dist_total[match_idx]:.6f}")
        
        ensure_dir(config.RESULTS_DIR)
        out_name = os.path.join(config.RESULTS_DIR, f"cspace_search_q{q_idx}.png")
        plot_search_analysis(q_audio, t_audio, dist_total, dist_fwd, dist_bwd, q_idx, match_idx, out_name)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python cspace_search.py <query_wav> <target_wav> <query_idx>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]))