import sys
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from espace_model import ESpace
import config

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_and_normalize_audio(path):
    print(f"Loading {path}...")
    audio, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    print(f"  -> Loaded {len(audio)} samples. Max Amp: {max_val:.4f}")
    return audio

def plot_spectrogram_with_waveform(states, audio_chunk, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), 
                                   gridspec_kw={'height_ratios': [1, 3]})
    
    # Waveform
    ax1.plot(audio_chunk, color='black', linewidth=0.8, alpha=0.7)
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Input Waveform")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.tick_params(labelbottom=False)

    # Spectrogram
    im = ax2.imshow(states.T, aspect='auto', origin='lower', cmap='inferno', interpolation='nearest')
    ax2.set_title(f"ESpace States ({config.RES_SIZE} Nodes, {config.LEAK_MIN} to {config.LEAK_MAX} Leaks)")
    ax2.set_ylabel("Node Index (Slow -> Fast)")
    ax2.set_xlabel("Time (Samples)")
    fig.colorbar(im, ax=ax2, label="Activation")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved visualization to {filename}")

def run_spectrogram(audio_full, sample_idx):
    # Set Seed
    np.random.seed(config.RANDOM_SEED)

    # Calculate Window
    half_window = config.WINDOW_SAMPLES // 2
    start_idx = max(0, sample_idx - half_window)
    end_idx = min(len(audio_full), sample_idx + half_window)
    query_audio = audio_full[start_idx:end_idx]
    
    print(f"Running Spectrogram: Processing window {start_idx}-{end_idx}")

    # Initialize Model
    model = ESpace(res_size=config.RES_SIZE, 
                   spectral_radius=config.SPECTRAL_RADIUS, 
                   sparsity=config.SPARSITY, 
                   leak_min=config.LEAK_MIN, 
                   leak_max=config.LEAK_MAX)
    
    # Forward Pass
    states = model.forward_sequence(query_audio)
    
    ensure_dir(config.RESULTS_DIR)
    output_path = os.path.join(config.RESULTS_DIR, f"spec_sample_{sample_idx}.png")
    plot_spectrogram_with_waveform(states, query_audio, output_path)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python esn_spectrogram.py <wav_path> <sample_index>")
        sys.exit(1)
        
    wav_path = sys.argv[1]
    try:
        query_sample = int(sys.argv[2])
    except ValueError:
        print("Error: Sample index must be an integer.")
        sys.exit(1)

    audio_data = load_and_normalize_audio(wav_path)
    run_spectrogram(audio_data, query_sample)