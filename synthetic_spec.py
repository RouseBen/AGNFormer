"""
Generate a synthetic spectral dataset (flux, wavelength, error) for testing
models that expect (X, wav, err) triples, e.g. NumericalDataset.

Wavelength grid mimics an SDSS-style setup: log10(lambda) spaced at a fixed
dispersion, covering ~3800-9200 A over `npix` pixels.
"""

import numpy as np

C_KMS = 299792.458


def make_wavelength_grid(npix=2500, lam_min=3800.0, lam_max=9200.0):
    """Shared rest-frame log10(wavelength) grid, common to every spectrum."""
    return np.linspace(np.log10(lam_min), np.log10(lam_max), npix)


def make_continuum(log_wav, n_spectra, rng):
    """Smooth pseudo-continuum: power law with random slope/normalisation."""
    lam = 10 ** log_wav
    slopes = rng.uniform(-2.0, 0.5, size=(n_spectra, 1))
    norms = rng.uniform(0.5, 5.0, size=(n_spectra, 1))
    ref_lam = 5000.0
    return norms * (lam[None, :] / ref_lam) ** slopes


def add_emission_lines(log_wav, flux, line_centres, rng, vwidth_kms=(300, 2000)):
    """Add Gaussian emission features at each line centre with random
    amplitude and a random velocity width (km/s -> sigma in log10 space)."""
    n_spectra, npix = flux.shape
    for _, center in line_centres:
        log_center = np.log10(center)
        amps = rng.uniform(0.0, 3.0, size=n_spectra)          # some lines weak/absent
        vwidths = rng.uniform(*vwidth_kms, size=n_spectra)     # km/s
        sigma_logw = (vwidths / C_KMS) / np.log(10)            # sigma in log10(lambda)

        diff = log_wav[None, :] - log_center                   # (1, npix)
        profile = np.exp(-0.5 * (diff / sigma_logw[:, None]) ** 2)
        flux += amps[:, None] * profile
    return flux


def generate_synthetic_spectra(
    n_spectra=1000,
    npix=2500,
    lam_min=3800.0,
    lam_max=9200.0,
    line_centres=None,
    noise_floor=0.02,
    seed=42,
):
    """
    Returns
    -------
    X   : (n_spectra, npix) noisy flux
    wav : (n_spectra, npix) log10(wavelength), same grid repeated per spectrum
    err : (n_spectra, npix) 1-sigma flux errors
    """
    rng = np.random.default_rng(seed)

    if line_centres is None:
        line_centres = [
            ("LYA", 1240.81),
            ("OI", 1305.53),
            ("Si IV", 1399.8),
            ("CIV", 1549.48),
            ("CIII", 1908.734),
            ("MgII", 2799.117),
            ("Hgamma", 4341.68),
            ("Hbeta", 4862.68),
            ("Halpha", 6564.61),
        ]

    log_wav = make_wavelength_grid(npix, lam_min, lam_max)
    wav = np.tile(log_wav, (n_spectra, 1))

    flux = make_continuum(log_wav, n_spectra, rng)
    flux = add_emission_lines(log_wav, flux, line_centres, rng)

    # Poisson-like errors: scale with sqrt(flux), plus a noise floor
    err = noise_floor * np.sqrt(np.abs(flux)) + noise_floor * 0.1
    flux = flux + rng.normal(0.0, err)

    # occasionally zero out edge pixels to mimic padded/missing coverage
    for i in range(n_spectra):
        n_pad = rng.integers(0, npix // 20)
        if n_pad > 0:
            flux[i, -n_pad:] = 0.0
            err[i, -n_pad:] = 0.0

    return flux.astype(np.float32), wav.astype(np.float32), err.astype(np.float32)

