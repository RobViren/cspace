import sys
import os
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
from cspace_model import CSpace
import config

def run(wav_path, idx):
    audio, _ = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
    max_val = np.max(np.abs(audio))
    if max_val > 0: audio = audio / max_val
    audio_t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0) # (1, Time)
    
    # Init Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CSpace(res_size=64).to(device)
    
    # Forward
    with torch.no_grad():
        # Output: (1, 256, Time) [R_F, I_F, R_B, I_B]
        states = model(audio_t.to(device)).cpu()
    
    # Slice Window
    hw = config.WINDOW_SAMPLES // 2
    s = max(0, idx - hw)
    e = min(states.shape[-1], idx + hw)
    
    # Extract Forward Magnitude
    # First 64 are RealF, Second 64 are ImagF
    res = 64
    real_f = states[0, 0:res, s:e]
    imag_f = states[0, res:2*res, s:e]
    mag_f = torch.sqrt(real_f**2 + imag_f**2).numpy()
    
    # Extract Backward Magnitude
    # Third 64 are RealB, Fourth 64 are ImagB
    real_b = states[0, 2*res:3*res, s:e]
    imag_b = states[0, 3*res:4*res, s:e]
    mag_b = torch.sqrt(real_b**2 + imag_b**2).numpy()
    
    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
    
    ax1.plot(audio[s:e], 'k', linewidth=0.5)
    ax1.set_title("Waveform")
    
    ax2.imshow(mag_f, aspect='auto', origin='lower', cmap='magma')
    ax2.set_title("Forward C-Space Magnitude")
    
    ax3.imshow(mag_b, aspect='auto', origin='lower', cmap='viridis')
    ax3.set_title("Backward C-Space Magnitude")
    
    if not os.path.exists(config.RESULTS_DIR): os.makedirs(config.RESULTS_DIR)
    plt.savefig(os.path.join(config.RESULTS_DIR, "cspace_viz.png"))
    print("Saved cspace_viz.png")

if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))