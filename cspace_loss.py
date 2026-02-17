import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from cspace_model import CSpace


class CSpaceLoss(nn.Module):
    def __init__(self, device, sr=24000, res_size=64, min_freq=10, max_freq=12000, 
                 decay_factor=0.999, high_freq_decay=0.6):
        super().__init__()
        
        # Multi-scale C-Space (3 scales: full, half, quarter resolution)
        self.cspace_full = CSpace(
            res_size=res_size, sr=sr, min_freq=min_freq, max_freq=max_freq,
            decay_factor=decay_factor, high_freq_decay=high_freq_decay
        ).to(device)
        
        self.cspace_half = CSpace(
            res_size=res_size, sr=sr//2, min_freq=min_freq, max_freq=max_freq//2,
            decay_factor=decay_factor, high_freq_decay=high_freq_decay
        ).to(device)
        
        self.cspace_quarter = CSpace(
            res_size=res_size, sr=sr//4, min_freq=min_freq, max_freq=max_freq//4,
            decay_factor=decay_factor, high_freq_decay=high_freq_decay
        ).to(device)
        
        # Multi-Scale Spectral Loss for timbre
        self.mels = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(sr, n_fft=s, hop_length=s//4, n_mels=64).to(device) 
            for s in [512, 1024, 2048]
        ])
        
        # Warmup/Warmdown samples to exclude (1000 samples = ~42ms at 24kHz)
        self.warmup = 1000
        
    def forward(self, pred, target):
        # === 1. Multi-Scale C-Space Loss (Same metric as search) ===
        
        # Full resolution
        p_c_full = self.cspace_full(pred)
        t_c_full = self.cspace_full(target)
        
        # Apply warmup/warmdown masking
        # Forward warmup: first 1000 samples invalid
        # Backward warmdown: last 1000 samples invalid (since backward has warmup at the end)
        p_c_full = p_c_full[..., self.warmup:-self.warmup]
        t_c_full = t_c_full[..., self.warmup:-self.warmup]
        
        # L2 norm over channel dimension (same as search distance)
        loss_cspace_full = torch.linalg.vector_norm(p_c_full - t_c_full, ord=2, dim=1).mean()
        
        # Half resolution (downsample 2x)
        pred_half = F.avg_pool1d(pred, kernel_size=2, stride=2)
        target_half = F.avg_pool1d(target, kernel_size=2, stride=2)
        
        p_c_half = self.cspace_half(pred_half)
        t_c_half = self.cspace_half(target_half)
        
        # Smaller warmup for downsampled (500 samples = same time duration)
        warmup_half = self.warmup // 2
        p_c_half = p_c_half[..., warmup_half:-warmup_half]
        t_c_half = t_c_half[..., warmup_half:-warmup_half]
        
        loss_cspace_half = torch.linalg.vector_norm(p_c_half - t_c_half, ord=2, dim=1).mean()
        
        # Quarter resolution (downsample 4x)
        pred_quarter = F.avg_pool1d(pred, kernel_size=4, stride=4)
        target_quarter = F.avg_pool1d(target, kernel_size=4, stride=4)
        
        p_c_quarter = self.cspace_quarter(pred_quarter)
        t_c_quarter = self.cspace_quarter(target_quarter)
        
        warmup_quarter = self.warmup // 4
        p_c_quarter = p_c_quarter[..., warmup_quarter:-warmup_quarter]
        t_c_quarter = t_c_quarter[..., warmup_quarter:-warmup_quarter]
        
        loss_cspace_quarter = torch.linalg.vector_norm(p_c_quarter - t_c_quarter, ord=2, dim=1).mean()
        
        # Combined multi-scale C-space
        loss_cspace = loss_cspace_full + 0.5 * loss_cspace_half + 0.25 * loss_cspace_quarter
        
        # === 2. Multi-Scale Spectral Loss (Timbre) ===
        loss_spec = 0.0
        for mel in self.mels:
            p_m = mel(pred).log1p()
            t_m = mel(target).log1p()
            loss_spec += F.l1_loss(p_m, t_m)
        
        # === 3. Time Domain L1 (Sample Accuracy Nudge) ===
        loss_time = F.l1_loss(pred, target)
        
        # === Weighted Sum ===
        # C-space leads (70%), spectral provides timbre (20%), time domain nudges (10%)
        return (10.0 * loss_cspace) + (3.0 * loss_spec) + (1.0 * loss_time)