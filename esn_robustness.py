import sys
import numpy as np
import librosa
from espace_model import ESpace
import config

# Define Test Configurations
# Format: (Name, Res Size, Seed, Sparsity)
EXPERIMENTS = [
    ("Baseline",       64,   42,    0.1),
    ("Tiny Net",       16,   42,    0.1),
    ("Large Net",      512,  42,    0.1),
    ("Seed A",         64,   1337,  0.1),
    ("Seed B",         64,   9999,  0.1),
    ("Ultra Sparse",   64,   42,    0.001), # Almost no recurrent connections
    ("Dense",          64,   42,    0.5),   # Heavy connections
]

def load_audio(path):
    print(f"Loading {path}...")
    audio, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio

def run_robustness_suite(wav_path, query_idx):
    audio = load_audio(wav_path)
    
    print(f"\n--- Running ESpace Robustness Suite ---")
    print(f"Target Sample Index: {query_idx}")
    print("-" * 95)
    print(f"{'Experiment Name':<15} | {'Nodes':<5} | {'Seed':<5} | {'Sparsity':<8} | {'Match Idx':<10} | {'Error':<5} | {'Sharpness':<10}")
    print("-" * 95)

    for name, res_size, seed, sparsity in EXPERIMENTS:
        # 1. Initialize specific model
        model = ESpace(res_size=res_size, 
                       spectral_radius=config.SPECTRAL_RADIUS, # Keep constant 0.99
                       sparsity=sparsity, 
                       leak_min=config.LEAK_MIN, 
                       leak_max=config.LEAK_MAX,
                       seed=seed)
        
        # 2. Forward Pass
        # We must re-run forward pass because changing seeds/nodes changes the vector space definition
        states = model.forward_sequence(audio)
        
        # 3. Get Query Vector
        if query_idx >= len(states):
            print(f"Index out of bounds for {name}")
            continue
            
        query_vec = states[query_idx]
        
        # 4. Search (Self-Similarity)
        # Calculate distance to all points
        # L2 norm of difference
        dists = np.linalg.norm(states - query_vec, axis=1)
        
        # 5. Find Match
        match_idx = np.argmin(dists)
        error = match_idx - query_idx
        
        # 6. Calculate Sharpness (Phase Sensitivity)
        # Distance at idx + 1. High value = steep local minima (Good phase lock).
        # Low value = flat minima (bad precision).
        if query_idx + 1 < len(dists):
            sharpness = dists[query_idx + 1]
        else:
            sharpness = dists[query_idx - 1]

        # 7. Log Result
        status_color = ""
        if error == 0:
            status = "PASS"
        else:
            status = f"FAIL ({error})"

        print(f"{name:<15} | {res_size:<5} | {seed:<5} | {sparsity:<8} | {match_idx:<10} | {error:<5} | {sharpness:.5f}")

    print("-" * 95)
    print("Sharpness = L2 Distance at 1 sample offset (Higher is tighter phase lock)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python esn_robustness.py <wav_path> <sample_index>")
        sys.exit(1)
        
    wav_path = sys.argv[1]
    try:
        idx = int(sys.argv[2])
    except ValueError:
        print("Error: Sample index must be an integer.")
        sys.exit(1)

    run_robustness_suite(wav_path, idx)