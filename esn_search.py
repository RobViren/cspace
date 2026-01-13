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
    # Load with native sampling rate, then resample if needed, or force config.SAMPLE_RATE
    # For robust testing, we enforce the rate defined in config
    audio, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    
    # Global Normalization
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    print(f"  -> Loaded {len(audio)} samples. Max Amp: {max_val:.4f}")
    return audio

def plot_search_results(query_audio, target_audio, dists, query_idx, match_idx, filename):
    window = config.WINDOW_SAMPLES  # Use the window from config (0.25s default, or 0.1s if you changed it)
    half_window = window // 2

    # 1. Prepare Query Window
    q_start = max(0, query_idx - half_window)
    q_end = min(len(query_audio), query_idx + half_window)
    q_chunk = query_audio[q_start:q_end]
    
    # 2. Prepare Match Window
    m_start = max(0, match_idx - half_window)
    m_end = min(len(target_audio), match_idx + half_window)
    m_chunk = target_audio[m_start:m_end]

    # 3. Prepare Distance Window (centered on match)
    # We want to see the "Loss Landscape" around the match
    d_chunk = dists[m_start:m_end]

    # Create Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    # --- Plot 1: Waveform Overlay ---
    # Create a common time axis relative to the center
    t_axis = np.arange(-half_window, half_window)
    # Handle edge cases where chunks might be shorter than window
    t_q = t_axis[:len(q_chunk)]
    t_m = t_axis[:len(m_chunk)]

    ax1.plot(t_q, q_chunk, color='black', linewidth=1.5, label='Query Waveform', alpha=0.8)
    ax1.plot(t_m, m_chunk, color='red', linestyle='--', linewidth=1.5, label='Matched Waveform', alpha=0.8)
    ax1.set_title(f"Waveform Alignment\nQuery Idx: {query_idx} | Match Idx: {match_idx} (Offset: {match_idx - query_idx})")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Distance Metric (The "Loss Landscape") ---
    # We plot the Euclidean distance of the states.
    t_d = t_axis[:len(d_chunk)]
    
    ax2.plot(t_d, d_chunk, color='blue', linewidth=1.5)
    ax2.set_title("State Space Euclidean Distance (Loss Landscape)")
    ax2.set_ylabel("L2 Distance")
    ax2.set_xlabel("Offset from Center (Samples)")
    
    # Mark the minimum
    min_dist = dists[match_idx]
    ax2.scatter([0], [min_dist], color='red', zorder=5)
    ax2.text(0, min_dist + (max(d_chunk)*0.05), f"{min_dist:.5f}", ha='center', color='red')

    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved search visualization to {filename}")

def run_search(query_path, target_path, query_idx):
    # Set Seed for determinism
    np.random.seed(config.RANDOM_SEED)

    # Load Audio
    query_audio = load_and_normalize_audio(query_path)
    if query_path == target_path:
        target_audio = query_audio
        print("Search Mode: Self-Similarity (Same File)")
    else:
        target_audio = load_and_normalize_audio(target_path)
        print("Search Mode: Cross-File Search")

    # Initialize Model
    print("Initializing ESpace...")
    model = ESpace(res_size=config.RES_SIZE, 
                   spectral_radius=config.SPECTRAL_RADIUS, 
                   sparsity=config.SPARSITY, 
                   leak_min=config.LEAK_MIN, 
                   leak_max=config.LEAK_MAX)

    # 1. Process Query
    # We need to process the file up to the query index to get the correct state context.
    # Since forward_sequence is fast (Rust), we just process the whole file for simplicity.
    print("Processing Query Audio...")
    query_states = model.forward_sequence(query_audio)
    
    if query_idx >= len(query_states):
        print(f"Error: Query index {query_idx} is out of bounds (Max: {len(query_states)})")
        sys.exit(1)

    query_vec = query_states[query_idx]

    # 2. Process Target
    # IMPORTANT: Reset state before processing the target file to ensure a clean "ruler" measurement
    model.backend.reset_state()
    print("Processing Target Audio...")
    target_states = model.forward_sequence(target_audio)

    # 3. Calculate Distances
    # Euclidean Distance between Query Vector and ALL Target vectors
    print("Calculating Distances...")
    # shape: (N_samples, N_nodes) - (N_nodes) -> broadcast
    diff = target_states - query_vec
    # L2 Norm across nodes axis
    dists = np.linalg.norm(diff, axis=1)

    # 4. Find Best Match
    match_idx = np.argmin(dists)
    print(f"Found Best Match at Sample: {match_idx} (Distance: {dists[match_idx]:.6f})")

    # 5. Visualize
    ensure_dir(config.RESULTS_DIR)
    out_name = f"search_q{query_idx}_match{match_idx}.png"
    output_path = os.path.join(config.RESULTS_DIR, out_name)
    
    plot_search_results(query_audio, target_audio, dists, query_idx, match_idx, output_path)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python esn_search.py <query_wav> <target_wav> <query_sample_index>")
        sys.exit(1)
        
    q_path = sys.argv[1]
    t_path = sys.argv[2]
    try:
        q_idx = int(sys.argv[3])
    except ValueError:
        print("Error: Sample index must be an integer.")
        sys.exit(1)

    run_search(q_path, t_path, q_idx)