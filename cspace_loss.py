import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from cspace_model import CSpace

class CSpaceLoss(nn.Module):
    def __init__(self, device, sr=24000, res_size=64):
        super().__init__()
        self.cspace = CSpace(res_size=res_size, sr=sr).to(device)
        
        # We also use Multi-Scale Spectral Loss as a "Timbre Check"
        # Pure phase loss sometimes ignores "buzz" high-freq artifacts that don't mess up phase much.
        self.mels = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(sr, n_fft=s, hop_length=s//4, n_mels=64).to(device) 
            for s in [512, 1024, 2048]
        ])

    def forward(self, pred, target):
        # 1. C-Space Holographic Loss (Phase & Structure)
        # States: (Batch, 256, Time) [RealF, ImagF, RealB, ImagB]
        p_c = self.cspace(pred)
        t_c = self.cspace(target)
        
        # L1 Loss on the Raw States (Forces Phase Locking)
        loss_phase = F.l1_loss(p_c, t_c)
        
        # 2. Magnitude Loss (Explicit Gain Matching)
        # Reconstruct Magnitude from the CSpace states
        # Fwd Mag
        res = p_c.shape[1] // 4
        p_mag_f = torch.sqrt(p_c[:, 0:res]**2 + p_c[:, res:2*res]**2 + 1e-9)
        t_mag_f = torch.sqrt(t_c[:, 0:res]**2 + t_c[:, res:2*res]**2 + 1e-9)
        loss_mag = F.l1_loss(p_mag_f, t_mag_f)
        
        # 3. Spectral Loss (Standard Timbre)
        loss_spec = 0.0
        for mel in self.mels:
            p_m = mel(pred).log1p()
            t_m = mel(target).log1p()
            loss_spec += F.l1_loss(p_m, t_m)
            
        # 4. Time Domain
        loss_time = F.l1_loss(pred, target)
        
        # Weighted Sum
        # We weigh Phase heavily because that's our unique value proposition
        return (10.0 * loss_phase) + (5.0 * loss_mag) + (1.0 * loss_spec) + (10.0 * loss_time)