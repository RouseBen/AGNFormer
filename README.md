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

Pre-trained weights from Rouse et al. 2026 are hosted on Zenodo: **[link to be added]**.

Download `transformer_weights.pt` and place it in the repository root (or point to its path) before running `example.py`.

## Usage

There are two ways to get a trained model:

### Option A: Use the pre-trained weights

In `example.py`, set:

```python
pretrain = False
weights = "transformer_weights.pt"   # path to the file downloaded from Zenodo
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

## Citation

If you use this code, please cite:

```
Rouse et al. 2026, in prep.
```

(Citation details to be updated on publication.)

## License

This project is licensed under the MIT License — see `LICENSE` for details.
