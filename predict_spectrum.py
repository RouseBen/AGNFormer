"""
Minimal example: run a trained AGNFormer model on a single real spectrum.

Replace the three arrays below (flux, wavelength, error) with your own data,
or point this script at files on disk (e.g. via np.load / np.loadtxt).

Expected input format (see README "Using your own data" section):
  - flux, error : 1D arrays of length `npix`, zero-padded wherever there is
    no coverage
  - wavelength  : 1D array of length `npix`, log10(Angstrom), rest-frame

Usage:
    python predict_spectrum.py
"""

import numpy as np
import torch

from base import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. Load your spectrum here. This placeholder uses a synthetic spectrum so
#    the script runs out of the box -- swap this block out for real data.
# ---------------------------------------------------------------------------
from synthetic_spec import generate_synthetic_spectra

X, wav, err = generate_synthetic_spectra(n_spectra=1, npix=2500)
flux = X[0]
wavelength = wav[0]
error = err[0]

npix = flux.shape[0]

# ---------------------------------------------------------------------------
# 2. Normalise exactly as during training (same formula as specmanip.py).
#    Skipping this step will produce meaningless predictions.
# ---------------------------------------------------------------------------
nonzero = flux != 0.0
div = np.sqrt(np.mean(flux[nonzero] ** 2))
flux_norm = np.clip(flux / div, -15, 15)
error_norm = np.clip(error / div, 0, 15)

# ---------------------------------------------------------------------------
# 3. Add a batch dimension and convert to tensors.
# ---------------------------------------------------------------------------
src = torch.tensor(flux_norm, dtype=torch.float32).unsqueeze(0).to(device)   # (1, npix)
errors = torch.tensor(error_norm, dtype=torch.float32).unsqueeze(0).to(device)  # (1, npix)
wavelength_t = torch.tensor(wavelength, dtype=torch.float32).unsqueeze(0).to(device)  # (1, npix)

# ---------------------------------------------------------------------------
# 4. Build the model with the SAME config the checkpoint was trained with,
#    then load the weights (see "Model weights" section of the README).
# ---------------------------------------------------------------------------
config = {"d_model": 256, "n_heads": 4, "n_layers": 8, "d_ff": 1024, "n_registers": 1}
wav_min, wav_max = float(wavelength.min()), float(wavelength.max())

model = Transformer(
    npix, npix, config["d_model"], config["n_heads"], config["n_layers"],
    config["d_ff"], npix, device, wav_min, wav_max, config["n_registers"],
)
model.load_state_dict(torch.load("weights.pt", map_location=device))
model.to(device)
model.eval()  # IMPORTANT: disables the noise-injection used during training
              # (base.py's Transformer.forward checks self.training, which is
              # only set correctly via model.eval()/model.train())

# ---------------------------------------------------------------------------
# 5. Run inference.
# ---------------------------------------------------------------------------
with torch.no_grad():
    mu, log_var = model(src, errors, wavelength_t, varian=True)

reconstructed_flux = mu.squeeze().cpu().numpy() * div
predicted_sigma = torch.exp(0.5 * log_var).squeeze().cpu().numpy() * div

print("Reconstructed flux (first 10 pixels):", reconstructed_flux[:10])
print("Predicted 1-sigma uncertainty (first 10 pixels):", predicted_sigma[:10])
