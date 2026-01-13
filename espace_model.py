import numpy as np
import espace_rs 
import config

class ESpace:
    def __init__(self, res_size=64, spectral_radius=0.99, sparsity=0.1, 
                 leak_min=1e-4, leak_max=1.0, seed=None):
        
        self.res_size = res_size
        
        # Determine Seed
        if seed is None:
            seed = config.RANDOM_SEED
        
        # Initialize the Rust backend
        # The Rust side handles:
        # 1. Sparse Matrix generation
        # 2. Spectral Radius scaling (via power iteration)
        # 3. Logarithmic leak rate distribution
        self.backend = espace_rs.RustESN(
            seed,
            res_size,
            sparsity,
            spectral_radius,
            leak_min,
            leak_max
        )

    def forward_sequence(self, audio):
        """
        Processes audio through the ESN.
        Audio should be a 1D numpy array (float32).
        Returns: (Samples, Nodes) array of states.
        """
        # Ensure float32 for Rust compatibility
        audio = audio.astype(np.float32)
        
        # Reset state before running a fresh isolated sample
        # (Unless we want state continuity, but for 'Search' we usually 
        # want the ruler to start fresh or have a warm-up period)
        self.backend.reset_state()
        
        # The Rust backend performs the sequential processing
        states = self.backend.forward_sequence(audio)
        
        return states

    def warmup(self, samples=1000):
        """Pushes zeros or noise to settle the state if needed."""
        dummy = np.zeros(samples, dtype=np.float32)
        self.backend.forward_sequence(dummy)