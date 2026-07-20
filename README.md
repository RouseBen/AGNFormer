# AGNFormer

Code accompanying **Rouse et al. 2026** (in prep) — *AGNFormer*, a transformer model for learning representations of active galactic nuclei (AGN) spectra via masked spectral reconstruction.

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

## Usage

Run the example training script (uses synthetic data, no real observations required):

```bash
python example.py
```

## Citation

If you use this code, please cite:

```
Rouse et al. 2026, in prep.
```

(Citation details to be updated on publication.)

## License

This project is licensed under the MIT License — see `LICENSE` for details.
