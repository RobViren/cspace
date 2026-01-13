# CSpace: Physically Grounded Audio Embeddings
### Solving Audio Phase Locking with Control Theory (No GANs Required)

**CSpace** (Cochlear Space) is an experimental framework that treats audio analysis not as a pattern recognition task (Perception), but as a measurement task (Control Theory).

By projecting audio into high-dimensional states of damped harmonic oscillators (via **Echo State Networks** and **Structured State Spaces**), we create a "Neural Ruler." This fixed coordinate system allows for:
1.  **Phase-Locked Similarity Search:** Precise alignment of waveforms even with offsets.
2.  **Holographic Loss:** A fully differentiable loss function that forces neural networks to learn phase without Adversarial (GAN) training.
3.  **Robust Embeddings:** Representations that survive quantization and noise.

---

## Installation

This project uses a mix of **Python (PyTorch)** and **Rust** (for the ESN reference implementation). It relies on `uv` for dependency management.

### Prerequisites
*   Python 3.10+
*   Rust (latest stable, via `rustup`)
*   CUDA (optional, for S4/Codec training)
*   `uv` (Universal Python Packaging)

### Setup

1.  **Install Python Dependencies:**
    ```bash
    uv add torch torchaudio torchcodec numpy scipy matplotlib librosa pyo3 datasets
    ```

2.  **Build the Rust Backend (ESpace):**
    This compiles the highly optimized ESN implementation via Maturin.
    ```bash
    uv tool install maturin
    maturin develop --release
    ```

3.  **Prepare Audio:**
    Place two test files in the `audio/` directory (e.g., `sample_1.wav`, `sample_2.wav`).

---

## Part 1: ESpace (Echo State Networks)
The initial exploration using random Recurrent Neural Networks as a liquid state machine.

### 1. Visualizing the "Neural Spectrogram"
Runs audio through a 64-node ESN with logarithmically distributed leak rates.
```bash
uv run esn_spectrogram.py ./audio/sample_1.wav 48000
```
*   **Output:** `results/spec_sample_48000.png`
*   **What to look for:** Horizontal bands corresponding to frequencies, but with phase-coherence visible in the activation patterns.

### 2. Similarity Search (The Ruler Test)
Takes a vector at sample `48000` and searches the rest of the file (or a target file) for the closest match using Euclidean distance.
```bash
uv run esn_search.py ./audio/sample_1.wav ./audio/sample_1.wav 48000
```
*   **Output:** `results/search_q48000_match_XXXX.png`
*   **Observation:** A sharp "V" shape in the distance plot indicates a precise basin of attraction for phase alignment.

### 3. Robustness Suite
Proves that the "Ruler" property holds across different network sizes, random seeds, and sparsity levels.
```bash
uv run esn_robustness.py ./audio/sample_1.wav 48000
```

---

## Part 2: CSpace (Holographic S4)
The evolution of the concept using **Complex Oscillators** and **FFT Convolution**. This moves from sequential processing (RNN) to parallel processing (LTI Systems) and introduces the **Bi-Directional Pincer** (Forward + Backward context).

### 1. Bi-Directional Visualization
Visualizes the Forward (Past Context) and Backward (Future Context) magnitude states.
```bash
uv run cspace_spectrogram.py ./audio/sample_1.wav 48000
```

### 2. Split-State Search
Demonstrates how the Forward and Backward states "triangulate" transient events.
```bash
uv run cspace_search.py ./audio/sample_1.wav ./audio/sample_1.wav 48000
```
*   **Observation:** The Forward loss drops *after* the event; the Backward loss drops *before* the event. The sum creates a perfect, steep gradient.

### 3. Gradient Topology (The Scientific Proof)
This compares the Loss Landscape of **Mel-Spectrograms** vs. **Multi-Scale Spectral** vs. **CSpace**.
```bash
uv run loss_comparision.py ./audio/sample_1.wav 48000
```
*   **Result:** `results/loss_topology_comparison.png`
*   **The Convex Basin:** While Spectral losses are flat or jagged (phase blind), CSpace produces a smooth, convex bowl pointing exactly to the zero-sample error.

---

## Part 3: Robustness & Stability
Testing the limits of the coordinate system under quantization and long-context drift.

### 1. Quantization Resilience (Int8)
Tests if the "Gradient Basin" survives being crushed into 8-bit integers using per-channel affine quantization.
```bash
uv run cspace_quantize.py ./audio/sample_1.wav 48000
```
*   **Result:** The Int8 loss landscape overlaps the Float32 landscape almost perfectly. This suggests CSpace embeddings are highly compressible while retaining semantic and phase fidelity.

### 2. Numerical Drift Test
Checks if the S4 states accumulate errors over long sequences (e.g., > 60 seconds).
```bash
uv run cspace_drift.py ./audio/sample_1.wav
```
*   **Observation:** Relative drift should be negligible (e.g., < 1e-6), proving CSpace is a stable mathematical transform, unlike drifting RNN states.

---

## Part 4: Neural Codec Training
Training a simple convolutional Autoencoder (Vocoder) using **only** the CSpace Holographic Loss. No Discriminators.

```bash
# Argument is the validation file to reconstruct at every checkpoint
uv run train_codec.py ./audio/sample_1.wav
```

*   **Process:** Streams data from LibriTTS (via HuggingFace).
*   **Logging:** Saves reconstructed audio every 500 steps to `results/training/`.
*   **Expectation:** Phase-locked, crisp audio within <2000 steps.

---

## Part 5: Inference & Generalization
Validating that the model learned the "Physics of Sound" and not just the statistics of speech.

### 1. Inference on Out-of-Distribution Audio (Music)
Runs the trained model on music or sound effects using an overlap-add chunking strategy for seamless processing.
```bash
# Usage: <checkpoint> <input> [output]
uv run inference.py results/training/codec_checkpoint.pth ./audio/music.wav output.wav
```

### 2. Fidelity Metrics
Calculates Time-Domain MSE, C-Space Error, and SI-SNR (Scale-Invariant Signal-to-Noise Ratio).
```bash
uv run measure_fidelity.py ./audio/music.wav output.wav
```
*   **Goal:** High SI-SNR (>20dB) on music, despite the model never seeing music during training. This confirms the model learned to use the CSpace Ruler as a generalized geometric projection.

---

## Project Structure

```text
.
├── Cargo.toml              # Rust configuration for ESpace
├── src/
│   └── lib.rs              # Rust ESN implementation (pyo3)
├── audio/                  # Test files
├── results/                # Generated plots and audio
├── config.py               # Global settings (SR=24000, Nodes=64, etc.)
├── espace_model.py         # Python wrapper for Rust ESN
├── cspace_model.py         # PyTorch S4 Complex Oscillator Module
├── cspace_loss.py          # The Hybrid Holographic Loss Module
├── esn_spectrogram.py      # ESN Visualization
├── esn_search.py           # ESN Similarity Search
├── esn_robustness.py       # ESN Config Validation
├── cspace_spectrogram.py   # S4 Visualization
├── cspace_search.py        # S4 Split-Search Visualization
├── cspace_robustness.py    # S4 Config Validation
├── cspace_quantize.py      # 8-bit Quantization Robustness
├── cspace_drift.py         # Long-Context Stability Test
├── loss_comparision.py     # Gradient Topology Experiment
├── train_codec.py          # Neural Vocoder Training Script
├── inference.py            # Overlap-Add Inference Script
└── measure_fidelity.py     # SI-SNR & MSE Validation Script
```

---

## Theory

**The "Ruler" vs. The "Perceptor"**
Most AI models (Transformers/CNNs) are "Perceptors"—they learn weights to extract features. CSpace is a "Ruler"—it projects audio into a fixed physical space (damped harmonic oscillation). Because the physics are fixed and differentiable, the "Ruler" provides a stable gradient for the model to learn against.

**The "Cursed" Architecture**
CSpace utilizes a bidirectional concatenation strategy (`[Real_Fwd, Imag_Fwd, Real_Bwd, Imag_Bwd]`). While mathematically redundant, this "Holographic" view ensures that a single time-step contains information about the entire history *and* future of the wave, forcing the model to generate physically consistent transients.