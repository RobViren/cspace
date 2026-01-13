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
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

# --- Quantization Functions ---

def quantize_fp16(tensor):
    """Simulate Float16 precision"""
    return tensor.half().float()

def quantize_int8_global(tensor):
    """Global Int8 Quantization (One scale for whole tensor)"""
    min_val = tensor.min()
    max_val = tensor.max()
    scale = 255.0 / (max_val - min_val + 1e-9)
    zero_point = -min_val * scale
    q = torch.clamp(torch.round(tensor * scale + zero_point), 0, 255).to(torch.uint8)
    dq = (q.float() - zero_point) / scale
    return dq

def quantize_int8_per_channel(tensor):
    """
    Per-Channel Int8 Quantization.
    Input shape: (Batch, Channels, Time)
    Calculates scale/zero_point for every Channel independently.
    """
    # Min/Max across Time dimension (dim=-1)
    # Output shape: (Batch, Channels, 1)
    min_val = tensor.min(dim=-1, keepdim=True).values
    max_val = tensor.max(dim=-1, keepdim=True).values
    
    # Calculate affine parameters per channel
    scale = 255.0 / (max_val - min_val + 1e-9)
    zero_point = -min_val * scale
    
    # Quantize (Broadcasts across time)
    q = torch.clamp(torch.round(tensor * scale + zero_point), 0, 255).to(torch.uint8)
    
    # Dequantize
    dq = (q.float() - zero_point) / scale
    return dq

# --- Main Test ---

def run_quantization_test(wav_path, center_idx):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load
    audio = load_audio_torch(wav_path).to(device)
    
    # Model
    model = CSpace(res_size=64).to(device)
    
    print("Generating Ground Truth States...")
    with torch.no_grad():
        states_gt = model(audio)
        
    # Define Tests
    tests = {
        "Float32 (Baseline)": states_gt.clone(),
        "Float16": quantize_fp16(states_gt),
        "Int8 (Global)": quantize_int8_global(states_gt),
        "Int8 (Per-Channel)": quantize_int8_per_channel(states_gt)
    }
    
    # Setup Window for Analysis
    win_size = 200
    half_win = win_size // 2
    s = center_idx - half_win
    e = center_idx + half_win
    
    plt.figure(figsize=(10, 6))
    print(f"\n--- Quantization Robustness (Idx: {center_idx}) ---")
    print(f"{'Precision':<20} | {'Error':<8} | {'Min Dist'}")
    print("-" * 50)
    
    for name, states_q in tests.items():
        # Window of target
        target_window = states_q[0, :, s:e]
        # Query vector (Center)
        query_vec = states_q[0, :, center_idx]
        
        # Distance
        diff = target_window - query_vec.unsqueeze(1)
        dists = torch.linalg.vector_norm(diff, ord=2, dim=0).cpu().numpy()
        
        # Normalize for Topology Comparison
        dists_norm = (dists - dists.min()) / (dists.max() - dists.min())
        
        # Check Error
        match_idx = np.argmin(dists)
        error = match_idx - half_win
        print(f"{name:<20} | {error:<8} | {dists.min():.6f}")
        
        # Plot Style
        if "Per-Channel" in name:
            style = '--'
            width = 2.0
            color = 'red'
        elif "Global" in name:
            style = ':'
            width = 1.5
            color = 'green'
        else:
            style = '-'
            width = 2.5
            color = None # Default
            
        x_axis = np.arange(-half_win, half_win)
        plt.plot(x_axis, dists_norm, label=name, linestyle=style, linewidth=width, color=color, alpha=0.8)

    plt.title(f"Loss Landscape: Per-Channel Quantization (Center: {center_idx})")
    plt.xlabel("Phase Offset (Samples)")
    plt.ylabel("Normalized Topology")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(0, color='red', alpha=0.2)
    
    ensure_dir(config.RESULTS_DIR)
    out = os.path.join(config.RESULTS_DIR, "quantization_topology.png")
    plt.savefig(out)
    print(f"\nSaved graph to {out}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python cspace_quantize.py <wav_path> <idx>")
        sys.exit(1)
    run_quantization_test(sys.argv[1], int(sys.argv[2]))