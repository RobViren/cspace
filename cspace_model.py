import torch
import torch.nn as nn
import torch.fft
import numpy as np
import config

class CSpace(nn.Module):
    def __init__(self, res_size=64, sr=24000, min_freq=10, max_freq=12000, 
                 decay_factor=0.999, high_freq_decay=0.6):
        super().__init__()
        
        self.res_size = res_size
        self.sr = sr
        
        # --- 1. Overshoot Frequencies (Logarithmic) ---
        # We go from 10Hz (Sub-bass) to Nyquist (12kHz)
        freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), res_size)
        
        # --- 2. Decays ---
        # We overshoot the decay range too. 
        # 0.999 is extremely resonant (rings for seconds).
        # 0.6 kills the signal almost instantly (transients).
        decays = np.linspace(decay_factor, high_freq_decay, res_size)
        
        # --- 3. Create Complex Kernels ---
        # Length: 1.0s to accommodate the slower decays (0.999 needs time to ring out)
        k_len = int(1.0 * sr)
        t = np.arange(k_len)
        
        # lambda = |r| * e^(i * theta)
        thetas = 2 * np.pi * freqs / sr
        eigenvalues = decays * np.exp(1j * thetas)
        
        # Kernel: K(t) = lambda^t
        # Shape: (Nodes, Time)
        kernels_np = eigenvalues[:, np.newaxis] ** t[np.newaxis, :]
        
        # Normalize energy so Bass doesn't dominate Treble
        # Division by sum of abs values essentially normalizes the gain
        kernels_np /= np.abs(kernels_np).sum(axis=1, keepdims=True)
        
        # Register as buffer (Complex64)
        self.register_buffer('kernels', torch.tensor(kernels_np, dtype=torch.complex64))

    def forward(self, audio):
        """
        Input: (Batch, Time)
        Returns: (Batch, Nodes * 4, Time) -> [RealF, ImagF, RealB, ImagB]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        # Ensure complex for FFT
        if not audio.is_complex():
            audio_c = audio.to(torch.complex64)
        else:
            audio_c = audio
            
        # Standard FFT Conv
        n_fft = audio.shape[-1] + self.kernels.shape[-1] - 1
        
        # Forward
        sig_f = torch.fft.fft(audio_c.unsqueeze(1), n=n_fft, dim=-1)
        ker_f = torch.fft.fft(self.kernels.unsqueeze(0), n=n_fft, dim=-1)
        out_f = torch.fft.ifft(sig_f * ker_f, n=n_fft, dim=-1)[..., :audio.shape[-1]]
        
        # Backward (Flip Audio, Convolve, Flip Result)
        audio_rev = torch.flip(audio_c, dims=[-1])
        sig_b = torch.fft.fft(audio_rev.unsqueeze(1), n=n_fft, dim=-1)
        # Reuse Kernel (assuming symmetry in time dynamics)
        out_b = torch.fft.ifft(sig_b * ker_f, n=n_fft, dim=-1)[..., :audio.shape[-1]]
        out_b = torch.flip(out_b, dims=[-1])
        
        # Flatten Complex to Real Channels
        # [RealF, ImagF, RealB, ImagB]
        return torch.cat([out_f.real, out_f.imag, out_b.real, out_b.imag], dim=1).float()