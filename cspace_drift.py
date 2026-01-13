import sys
import torch
import numpy as np
import librosa
from cspace_model import CSpace
import config

def run_drift_test(wav_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Long-Context Drift Test ---")
    
    # Load
    audio, _ = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
    audio = audio / np.max(np.abs(audio))
    audio_t = torch.tensor(audio).float().to(device)
    
    # Calculate repeats to get approx 60 seconds
    # 60 seconds is plenty to prove S4 stability (Impulse decays in < 1s)
    # 20 minutes (previous try) causes OOM due to FFT buffer size
    target_len_sec = 60.0
    file_len_sec = len(audio) / config.SAMPLE_RATE
    repeats = int(target_len_sec / file_len_sec) + 1
    
    print(f"Input Length: {file_len_sec:.2f}s")
    print(f"Tiling {repeats} times to reach > {target_len_sec}s...")
    
    long_audio = audio_t.repeat(repeats).unsqueeze(0)
    total_samples = long_audio.shape[-1]
    
    print(f"Processing {total_samples} samples ({total_samples/24000:.2f}s)...")
    
    model = CSpace(res_size=64).to(device)
    
    # Process
    try:
        with torch.no_grad():
            states = model(long_audio) # (1, 256, Total_Time)
    except torch.cuda.OutOfMemoryError:
        print("\nERROR: Out of Memory. The sequence is still too long for naive FFT convolution.")
        print("For production processing of hour-long files, you must use Chunked (Overlap-Save) convolution.")
        print("For this theoretical Drift Test, try a shorter duration.")
        return

    # Compare 1st repetition vs Last repetition
    segment_len = len(audio)
    
    # Pick a point in the middle of the sample
    local_idx = segment_len // 2
    
    # Index in first clip
    idx_start = local_idx
    
    # Index in last clip
    idx_end = local_idx + ((repeats - 1) * segment_len)
    
    print(f"\nComparing State at T={idx_start/24000:.2f}s vs T={idx_end/24000:.2f}s")
    
    vec_start = states[0, :, idx_start]
    vec_end = states[0, :, idx_end]
    
    # Measure Drift
    diff = vec_start - vec_end
    dist = torch.linalg.vector_norm(diff, ord=2).item()
    mag = torch.linalg.vector_norm(vec_start, ord=2).item()
    rel_error = dist / (mag + 1e-9)
    
    print("-" * 40)
    print(f"Vector Magnitude: {mag:.5f}")
    print(f"Absolute Drift:   {dist:.5f}")
    print(f"Relative Drift:   {rel_error:.9f}")
    print("-" * 40)
    
    if rel_error < 1e-4:
        print("RESULT: STABLE (No Accumulation of Error)")
    else:
        print("RESULT: UNSTABLE (Numerical Drift Detected)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cspace_drift.py <wav_path>")
        sys.exit(1)
    run_drift_test(sys.argv[1])