"""
Example: extract and plot attention matrices from a trained AGNFormer model.

Setting `probs=True` when constructing the `Transformer` makes forward() return
an extra `all_attn_probs` list: one attention-weight tensor per encoder layer,
each of shape (batch, heads, seq_len [+ n_registers], seq_len [+ n_registers]).
The extra row/column (if `n_registers > 0`) corresponds to the learnable
register token(s), which attend to (and are attended to by) every spectral
pixel -- `plot_all_attention` below trims the last row/column, assuming a
single register token (the default, `n_registers=1`).

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


def plot_all_attention(all_attn_probs, wavelength, count, vmax=None, n_ticks=6):
    """
    Plot attention probabilities for all layers and heads in a grid.

    Parameters
    ----------
    all_attn_probs : list[torch.Tensor]
        List of [B, n_heads, S, S] tensors (one per encoder layer).
    wavelength : torch.Tensor
        Wavelength values corresponding to the sequence positions, in
        log10(Angstrom), shape (B, S) -- the same tensor passed to the model.
    count : int or str
        Identifier used in the saved filename (e.g. spectrum index).
    vmax : float, optional
        Maximum value for color scale normalisation (useful for consistency
        across multiple figures).
    n_ticks : int, optional
        Number of wavelength ticks on each axis.
    """
    n_layers = len(all_attn_probs)
    n_heads = all_attn_probs[0].shape[1]
    seq_len = all_attn_probs[0].shape[-1]

    wavelength = 10 ** wavelength[0, :seq_len].cpu().numpy()
    tick_indices = np.linspace(0, len(wavelength) - 1, n_ticks, dtype=int)
    tick_labels = np.round(wavelength[tick_indices], 1)

    fig, axes = plt.subplots(
        n_layers, n_heads, figsize=(3 * n_heads, 3 * n_layers), sharey=True, sharex=True
    )

    for i, layer_attn in enumerate(all_attn_probs):
        for j in range(n_heads):
            ax = axes[i, j] if n_layers > 1 else axes[j]

            attn = layer_attn[0, j].detach().cpu().numpy()  # [S, S]

            # Drop the last row/column, i.e. the register token (assumes n_registers=1)
            im = ax.imshow(
                np.log10(attn[:-1, :-1]), cmap="viridis", origin="lower",
                aspect="auto", vmin=-4, vmax=vmax,
            )

            ax.set_title(f"Head {j + 1}", fontsize=15) if i == 0 else None

            if i == 0:
                ax.set_title(f"Head {j + 1}", fontsize=15)
                ax.set_xticks([])
                ax.set_xticklabels([])
            elif i == n_layers - 1:
                ax.set_xticks(tick_indices)
                ax.set_xticklabels(tick_labels, rotation=45, fontsize=12)
                ax.set_xlabel("Wavelength (Å)", fontsize=15)
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(f"Layer {i + 1}\nWavelength (Å)", fontsize=15)
                ax.set_yticks(tick_indices)
                ax.set_yticklabels(tick_labels, fontsize=12)
            else:
                ax.set_yticks([])

    fig.tight_layout()
    plt.savefig(f"assets/attn_plot_{count}.png", dpi=200)
    plt.close()


if __name__ == "__main__":
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
        probs=True,  # <-- key flag: makes forward() also return attention matrices
    )
    model.load_state_dict(torch.load("weights.pt", map_location=device))
    model.to(device)
    model.eval()

    # --- 4. Run inference; all_attn_probs has one entry per encoder layer ---
    with torch.no_grad():
        mu, log_var, all_attn_probs = model(src, errors, wavelength_t, varian=True)

    # --- 5. Plot every layer/head in a grid, saved to assets/attn_plot_<count>.png ---
    plot_all_attention(all_attn_probs, wavelength_t, count=0)
