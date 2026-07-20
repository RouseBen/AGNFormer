"""
Example: extract and plot attention matrices from a trained AGNFormer model.

Setting `probs=True` when constructing the `Transformer` makes forward() return
an extra `all_attn_probs` list: one attention-weight tensor per encoder layer,
each of shape (batch, heads, seq_len [+ n_registers], seq_len [+ n_registers]).
The extra rows/columns (if `n_registers > 0`) correspond to the learnable
register tokens, which attend to (and are attended to by) every spectral pixel.

This script reuses the normalisation/loading steps from predict_spectrum.py --
see that file for more detail on each step.

Usage:
    python plot_attention.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from base import Transformer
from synthetic_spec import generate_synthetic_spectra

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Load a spectrum (placeholder synthetic data -- swap for real data) ---
X, wav, err = generate_synthetic_spectra(n_spectra=1, npix=2500)
flux, wavelength, error = X[0], wav[0], err[0]
npix = flux.shape[0]

# --- 2. Normalise exactly as during training ---
nonzero = flux != 0.0
div = np.sqrt(np.mean(flux[nonzero] ** 2))
flux_norm = np.clip(flux / div, -15, 15)
error_norm = np.clip(error / div, 0, 15)

src = torch.tensor(flux_norm, dtype=torch.float32).unsqueeze(0).to(device)
errors = torch.tensor(error_norm, dtype=torch.float32).unsqueeze(0).to(device)
wavelength_t = torch.tensor(wavelength, dtype=torch.float32).unsqueeze(0).to(device)

# --- 3. Build the model with probs=True so attention weights are returned ---
config = {"d_model": 256, "n_heads": 4, "n_layers": 8, "d_ff": 1024, "n_registers": 1}
wav_min, wav_max = float(wavelength.min()), float(wavelength.max())

model = Transformer(
    npix, npix, config["d_model"], config["n_heads"], config["n_layers"],
    config["d_ff"], npix, device, wav_min, wav_max, config["n_registers"],
    probs=True,  # <-- this is the key flag: makes forward() also return attention matrices
)
model.load_state_dict(torch.load("weights.pt", map_location=device))
model.to(device)
model.eval()

# --- 4. Run inference; all_attn_probs is a list with one entry per encoder layer ---
with torch.no_grad():
    mu, log_var, all_attn_probs = model(src, errors, wavelength_t, varian=True)

# --- 5. Plot the attention matrix for one layer / head ---
layer_idx = -1   # last encoder layer
head_idx = 0

attn = all_attn_probs[layer_idx][0, head_idx].cpu().numpy()  # (seq_len [+ n_registers], seq_len [+ n_registers])

plt.figure(figsize=(6, 6))
plt.imshow(attn, cmap="viridis", aspect="auto")
plt.colorbar(label="Attention weight")
plt.xlabel("Key position (pixel index, register tokens last)")
plt.ylabel("Query position (pixel index, register tokens last)")
plt.title(f"Attention matrix -- layer {layer_idx}, head {head_idx}")
plt.tight_layout()
plt.savefig("assets/attention_example.png", dpi=150)
plt.show()
