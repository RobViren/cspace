import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import matplotlib.pyplot as plt
import torchaudio
from cspace_model import CSpace


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_audio_torch(path, sr=24000):
    print(f"Loading {path}...")
    audio, orig_sr = librosa.load(path, sr=sr, mono=True)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # (1, Time)


def compute_cspace_distance(audio, target_idx, model, device):
    """Compute C-Space L2 distance at each offset, same as search metric"""
    with torch.no_grad():
        states = model(audio.to(device))  # (1, Channels, Time)
        
        # Extract query vector at target index
        q_vec = states[0, :, target_idx]  # (Channels,)
        
        # Compute L2 distance over channel dimension for each time step
        diff = states[0] - q_vec.unsqueeze(1)  # (Channels, Time)
        distances = torch.linalg.vector_norm(diff, ord=2, dim=0).cpu().numpy()
        
    return distances


def compute_mss_distance(audio, target_idx, device, sr=24000):
    """Multi-Scale Spectral Distance (standard perceptual metric)"""
    # Use standard mel spectrogram settings
    mel_transforms = [
        torchaudio.transforms.MelSpectrogram(sr, n_fft=n, hop_length=n//4, n_mels=64).to(device)
        for n in [512, 1024, 2048]
    ]
    
    with torch.no_grad():
        audio = audio.to(device)
        T = audio.shape[-1]
        
        # Compute spectrograms for full audio
        specs = []
        for mel in mel_transforms:
            spec = mel(audio).log1p()  # (1, n_mels, time_frames)
            specs.append(spec)
        
        # For each offset, compute spectral distance in a local window
        # We need to align spectrograms around target point
        distances = np.zeros(T)
        
        # Convert sample index to spectrogram frame index (approximate)
        # hop_length for different scales
        hop_lengths = [128, 256, 512]  # n_fft // 4
        
        for t in range(T):
            dist = 0.0
            for spec, hop in zip(specs, hop_lengths):
                # Convert sample indices to frame indices
                target_frame = target_idx // hop
                t_frame = t // hop
                
                # Extract local window (±5 frames)
                window = 5
                t_start = max(0, t_frame - window)
                t_end = min(spec.shape[-1], t_frame + window)
                target_start = max(0, target_frame - window)
                target_end = min(spec.shape[-1], target_frame + window)
                
                # Pad if needed to match shapes
                t_window = spec[0, :, t_start:t_end]
                target_window = spec[0, :, target_start:target_end]
                
                min_len = min(t_window.shape[-1], target_window.shape[-1])
                if min_len > 0:
                    dist += F.l1_loss(t_window[:, :min_len], target_window[:, :min_len]).item()
            
            distances[t] = dist
    
    return distances


def compute_waveform_l1_distance(audio, target_idx, window=100):
    """Waveform L1 distance in local window"""
    audio_np = audio.squeeze().numpy()
    T = len(audio_np)
    distances = np.zeros(T)
    
    # Extract target window
    target_start = max(0, target_idx - window)
    target_end = min(T, target_idx + window)
    target_window = audio_np[target_start:target_end]
    
    for t in range(T):
        # Extract window at offset t
        t_start = max(0, t - window)
        t_end = min(T, t + window)
        t_window = audio_np[t_start:t_end]
        
        # Compute L1 distance (handle different window sizes)
        min_len = min(len(t_window), len(target_window))
        if min_len > 0:
            distances[t] = np.mean(np.abs(t_window[:min_len] - target_window[:min_len]))
        else:
            distances[t] = np.inf
    
    return distances


def plot_comparison(audio, distances_dict, target_idx, config_info, output_path):
    """Plot loss topology comparison"""
    window = 100
    audio_np = audio.squeeze().numpy()
    
    # Slice window around target
    plot_start = max(0, target_idx - window)
    plot_end = min(len(audio_np), target_idx + window)
    
    x_axis = np.arange(plot_start, plot_end) - target_idx  # Centered at 0
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # === Plot 1: Waveform ===
    axes[0].plot(x_axis, audio_np[plot_start:plot_end], 'k', linewidth=1.0)
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Target')
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Waveform")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # === Plot 2: C-Space Loss ===
    cspace_slice = distances_dict['cspace'][plot_start:plot_end]
    axes[1].plot(x_axis, cspace_slice, 'b-', linewidth=2.0, label='C-Space (L2)')
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[1].set_ylabel("Distance")
    axes[1].set_title("C-Space Loss Topology")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Mark minimum
    min_idx = np.argmin(cspace_slice)
    axes[1].scatter([x_axis[min_idx]], [cspace_slice[min_idx]], c='red', s=100, zorder=10)
    
    # === Plot 3: Multi-Scale Spectral Loss ===
    mss_slice = distances_dict['mss'][plot_start:plot_end]
    axes[2].plot(x_axis, mss_slice, 'g-', linewidth=2.0, label='MSS (L1)')
    axes[2].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[2].set_ylabel("Distance")
    axes[2].set_title("Multi-Scale Spectral Loss Topology")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Mark minimum
    min_idx = np.argmin(mss_slice)
    axes[2].scatter([x_axis[min_idx]], [mss_slice[min_idx]], c='red', s=100, zorder=10)
    
    # === Plot 4: Waveform L1 Loss ===
    l1_slice = distances_dict['l1'][plot_start:plot_end]
    axes[3].plot(x_axis, l1_slice, 'm-', linewidth=2.0, label='Waveform L1')
    axes[3].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[3].set_ylabel("Distance")
    axes[3].set_xlabel("Offset from Target (samples)")
    axes[3].set_title("Waveform L1 Loss Topology")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Mark minimum
    min_idx = np.argmin(l1_slice)
    axes[3].scatter([x_axis[min_idx]], [l1_slice[min_idx]], c='red', s=100, zorder=10)
    
    # Add configuration info as text box
    config_text = "\n".join([f"{k}: {v}" for k, v in config_info.items()])
    fig.text(0.02, 0.98, config_text, fontsize=9, family='monospace',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def analyze_sharpness(distances, target_idx):
    """Compute basin sharpness metrics"""
    # Sharpness = distance at ±1 sample offset
    left_sharp = distances[target_idx - 1] if target_idx > 0 else np.inf
    right_sharp = distances[target_idx + 1] if target_idx < len(distances) - 1 else np.inf
    sharpness = (left_sharp + right_sharp) / 2.0
    
    # Basin width = samples until distance doubles from minimum
    min_dist = distances[target_idx]
    threshold = min_dist * 2.0
    
    # Search left
    left_width = 0
    for i in range(target_idx - 1, -1, -1):
        if distances[i] > threshold:
            break
        left_width += 1
    
    # Search right
    right_width = 0
    for i in range(target_idx + 1, len(distances)):
        if distances[i] > threshold:
            break
        right_width += 1
    
    basin_width = left_width + right_width
    
    return sharpness, basin_width


def run_comparison(wav_path, target_idx, 
                   res_size=64, min_freq=10, max_freq=12000, 
                   decay_factor=0.999, high_freq_decay=0.6):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load audio
    audio = load_audio_torch(wav_path)
    
    # Initialize C-Space model
    print("Initializing C-Space model...")
    cspace_model = CSpace(
        res_size=res_size,
        sr=24000,
        min_freq=min_freq,
        max_freq=max_freq,
        decay_factor=decay_factor,
        high_freq_decay=high_freq_decay
    ).to(device)
    
    # Compute distances
    print("Computing C-Space distances...")
    cspace_dists = compute_cspace_distance(audio, target_idx, cspace_model, device)
    
    print("Computing Multi-Scale Spectral distances...")
    mss_dists = compute_mss_distance(audio, target_idx, device)
    
    print("Computing Waveform L1 distances...")
    l1_dists = compute_waveform_l1_distance(audio, target_idx, window=100)
    
    distances_dict = {
        'cspace': cspace_dists,
        'mss': mss_dists,
        'l1': l1_dists
    }
    
    # Analyze sharpness
    print("\n=== Basin Analysis ===")
    for name, dists in distances_dict.items():
        sharp, width = analyze_sharpness(dists, target_idx)
        print(f"{name.upper():<10} | Sharpness: {sharp:.6f} | Basin Width: {width} samples")
    
    # Configuration info for plot
    config_info = {
        'Target Index': target_idx,
        'Sample Rate': '24000 Hz',
        'Nodes': res_size,
        'Freq Range': f'{min_freq}-{max_freq} Hz',
        'Decay Range': f'{decay_factor}-{high_freq_decay}',
        'Window': '±100 samples',
        'Device': str(device)
    }
    
    # Plot
    ensure_dir('results')
    output_path = f'results/loss_comparison_idx{target_idx}_nodes{res_size}.png'
    plot_comparison(audio, distances_dict, target_idx, config_info, output_path)
    
    print(f"\nDone! Check {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_loss_topology.py <audio_path> <target_idx> [res_size] [min_freq] [max_freq] [decay_low] [decay_high]")
        print("Example: python compare_loss_topology.py audio.wav 48000 64 10 12000 0.999 0.6")
        sys.exit(1)
    
    wav_path = sys.argv[1]
    target_idx = int(sys.argv[2])
    
    # Optional parameters
    res_size = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    min_freq = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    max_freq = int(sys.argv[5]) if len(sys.argv) > 5 else 12000
    decay_low = float(sys.argv[6]) if len(sys.argv) > 6 else 0.999
    decay_high = float(sys.argv[7]) if len(sys.argv) > 7 else 0.6
    
    run_comparison(wav_path, target_idx, res_size, min_freq, max_freq, decay_low, decay_high)