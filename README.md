# AGNFormer

Code accompanying **Rouse et al. 2026** (in prep) — *AGNFormer*, reconstruction of active galactic nuclei spectra using a probabilistic transformer model.

## Overview
AGNFormer (`base.py`) is a transformer encoder trained to reconstruct masked or corrupted regions of an input spectrum (flux + error vs. wavelength), predicting both a mean flux (`mu`) and an uncertainty (`log_var`) at each pixel via a Gaussian negative log-likelihood loss. Wavelength is encoded with a continuous sinusoidal positional encoding (`WavelengthPositionalEncodingVaswaniEquivalent`), and learnable register tokens are prepended to the sequence to accumulate global spectral information.

Several masking strategies are provided in `specmanip.py` to corrupt spectra during training/evaluation:

- `FixedMask` — masks fixed wavelength windows around known emission lines (`line_centres.py`)
- `RandomMaskLines` — randomly masks a fraction of pixels
- `RandomMaskChunks` — randomly masks contiguous chunks of the spectrum
- `FractionalMask` — masks a fixed fraction from the blue or red end of the spectrum

## Repository structure

| File | Description |
|---|---|
| `base.py` | Core model: multi-head attention, transformer encoder layers, embedding layers, and the full `Transformer` model with variational (mean + log-variance) output heads. |
| `example.py` | End-to-end example: generates synthetic spectra, builds train/val/test splits, and runs the pre-training loop. |
| `functions.py` | `NLLloss` — Gaussian negative log-likelihood / MSE loss used for training. |
| `specmanip.py` | Spectrum masking and normalisation utilities used during training. |
| `synthetic_spec.py` | Generates synthetic AGN-like spectra (continuum + emission lines + noise) for testing the pipeline without real data. |
| `line_centres.py` | Table of rest-frame broad emission-line centres (from the SDSS line list) used by `FixedMask`. |
| `predict_spectrum.py` | Minimal example: load a trained model and run it on a single spectrum to get a reconstruction. |
| `plot_attention.py` | Example: extract attention matrices (`probs=True`) from a trained model and plot them. |

## Requirements

- Python 3.12
- torch
- numpy
- pandas
- scikit-learn
- matplotlib
- astropy
- h5py
- scipy
- [`scheduled`](https://github.com/fabian-sp/lr-scheduling) — provides the `CosineSchedule` / `WSDSchedule` learning-rate schedules used in `example.py` and `base.py`; install with:
  ```bash
  pip install git+https://github.com/fabian-sp/lr-scheduling.git
  ```

See `requirements.txt` for a pinned-free install list (`pip install -r requirements.txt`).

## Model weights

Pre-trained weights from Rouse et al. 2026 are hosted on Zenodo (record to be published — link will be updated here once live).

Once published, download `weights.pt` into the repository root with:

```bash
curl -L -o weights.pt "https://zenodo.org/records/<RECORD_ID>/files/weights.pt?download=1"
```


## Usage

There are two ways to get a trained model:

### Option A: Use the pre-trained weights

In `example.py`, set:

```python
pretrain = False
weights = "weights.pt"   # path to the file downloaded from Zenodo
```

The `config` dict (`d_model`, `n_heads`, `n_layers`, `d_ff`, `n_registers`) must match the architecture the released weights were trained with — don't change these unless you're training your own model from scratch (Option B).

### Option B: Pre-train from scratch

Leave the default in `example.py`:

```python
pretrain = True
```

This skips loading any checkpoint and trains a fresh model. You're free to change `config` in this case.

Then run:

```bash
python example.py
```

### Using your own data

By default `example.py` calls `generate_synthetic_spectra()` to create toy spectra so the pipeline runs out of the box. To train or fine-tune on real spectra, replace that call with your own data loading, providing three arrays of matching shape `(n_spectra, npix)`:

- `X` — flux, one row per spectrum (pad missing/edge pixels with `0.0`)
- `wav` — wavelength grid in `log10(Angstrom)`, rest-frame — the same units used by `line_centres.py` and `FixedMask`
- `err` — 1-sigma flux uncertainty per pixel (`0.0` wherever `X` is padded)

Every spectrum should share the same `npix` (pad/truncate/interpolate onto a common grid beforehand). See `synthetic_spec.py` for a worked example of the expected array layout.

### Predicting on a single spectrum

`predict_spectrum.py` is a minimal, runnable example showing how to feed one spectrum through a trained model and get a reconstruction back:

```bash
python predict_spectrum.py
```

It covers the steps needed to use your own data at inference time: normalising the spectrum the same way as during training, adding the batch dimension, building the model with the checkpoint's config, loading `weights.pt`, and calling `model.eval()` before running inference (skipping `.eval()` is a common mistake — the model injects training-time noise unless it's in eval mode). Swap out the placeholder synthetic spectrum near the top of the script for your own flux/wavelength/error arrays.

### Visualising attention

Passing `probs=True` when constructing `Transformer` makes `forward()` return an extra `all_attn_probs` value: a list with one attention-weight tensor per encoder layer, each of shape `(batch, heads, seq_len [+ n_registers], seq_len [+ n_registers])`. This lets you inspect which wavelength regions the model attends to when reconstructing a spectrum.

```python
model = Transformer(..., probs=True)
...
mu, log_var, all_attn_probs = model(src, errors, wavelength, varian=True)
attn = all_attn_probs[-1][0, 0]  # last layer, batch item 0, head 0 -> (seq_len [+ n_registers], seq_len [+ n_registers])
```

Note that if `n_registers > 0`, the attention matrix includes extra rows/columns for the register tokens (appended after the spectral pixels), which attend to and are attended to by every pixel.

`plot_attention.py` is a runnable example that builds a model with `probs=True`, loads `weights.pt`, and saves a heatmap of one layer/head's attention matrix to `assets/attention_example.png`:

```bash
python plot_attention.py
```

Example output:

![Example attention matrix](assets/attention_example.png)

*(placeholder — replace `assets/attention_example.png` with a real figure from a trained model)*

## Citation

If you use this code, please cite:

```
Rouse et al. 2026, in prep.
```

(Citation details to be updated on publication.)

## License

This project is licensed under the MIT License — see `LICENSE` for details.
