import numpy as np

import torch
import torch.nn as nn

from line_centres import centres

C_KMS = 299792.458

def build_mask_ranges(line_centres, width_kms, merge_overlapping=True):
    """
    line_centres : list of (name, wavelength_angstrom)
    width_kms    : full mask width in km/s, applied around every line centre
                   (delta_lambda = center * width_kms / c)
    merge_overlapping : merge ranges that end up overlapping (e.g. close doublets)

    Returns list of (log10_low, log10_high) tuples.
    """
    ranges = []
    for name, center in line_centres:
        half = center * (width_kms / C_KMS) / 2
        ranges.append((np.log10(center - half), np.log10(center + half)))

    ranges.sort(key=lambda r: r[0])

    if merge_overlapping:
        merged = [ranges[0]]
        for low, high in ranges[1:]:
            prev_low, prev_high = merged[-1]
            if low <= prev_high:
                merged[-1] = (prev_low, max(prev_high, high))
            else:
                merged.append((low, high))
        ranges = merged

    return ranges


class FixedMask(nn.Module):
    def __init__(self, seq_len, random_seed = 48, line_centres = centres, mask_width_kms = 10000):
        super().__init__()
        self.seed = random_seed
        self.seq_len = seq_len
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.mask_ranges = build_mask_ranges(line_centres, mask_width_kms)

    def forward(self, x, w,err=None):

        batch_size = x.shape[0]
        padded_x = x[:,0:self.seq_len].clone()
        padded_w = w[:,0:self.seq_len].clone()

        if err is not None:
            padded_e = err[:,0:self.seq_len].clone()


        src_mask = (padded_x != 0).unsqueeze(1).unsqueeze(2)
        batch_mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool, device=x.device)


        for low, high in self.mask_ranges:
            batch_mask |= (padded_w >= low) & (padded_w <= high)


        # --- apply mask to flux ---
        masked_x = padded_x.clone()          # one clone for the whole batch
        masked_x[batch_mask] = 0.0

        if err is not None:
            masked_e = padded_e.clone()
            masked_e_broad = padded_e.clone()
            masked_e[batch_mask] = 0.0
            masked_e_broad[~batch_mask] = 0.0

        # --- vectorised normalisation (no per-sample loop) ---
        # count non-zero per sample: (batch,)
        non_zero_mask = padded_x != 0.0 #masked_x #!= 0.0
        count = non_zero_mask.sum(dim=1, keepdim=True).clamp(min=1)  # avoid div/0

        # sum of squares of non-zero elements per sample
        flux_sq_sum = (masked_x ** 2 * non_zero_mask).sum(dim=1, keepdim=True)
        div = torch.sqrt(flux_sq_sum / count)                         # (batch, 1)

        # normalise — div broadcasts across seq_len dimension
        normalised_flux = padded_x / div
        normalised_mask = masked_x / div
        normalised_err = padded_e / div
        normalised_err_mask = masked_e / div
        normed_br_err_mask = masked_e_broad / div

        # --- clip outliers vectorised ---
        normalised_flux = normalised_flux.clamp(-15, 15)
        normalised_mask = normalised_mask.clamp(-15, 15)
        normalised_err = normalised_err.clamp(0, 15)
        normalised_err_mask = normalised_err_mask.clamp(0, 15)
        normed_br_err_mask = normed_br_err_mask.clamp(0,15)
        # --- src_mask vectorised ---
        src_mask = (padded_x != 0).unsqueeze(1).unsqueeze(2)

        # return order:
        # masked spectrum (input), whole spectrum (output)
        # error spectrum, masked error spec (input), broad line err spec (loss)
        # padded wavelength, src mask, batch mask
        return normalised_mask, normalised_flux, normalised_err, normalised_err_mask,normed_br_err_mask, padded_w, src_mask, batch_mask

class FractionalMask(nn.Module):
    def __init__(self, seq_len, random_seed = 48, red = True, frac = 0.5):
        super().__init__()
        self.seed = random_seed
        self.seq_len = seq_len
        self.red = red
        self.frac = frac
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)


    def forward(self, x, w,err):
        batch_size = x.shape[0]
        padded_x = x[:,0:self.seq_len].clone()
        padded_w = w[:,0:self.seq_len].clone()

        padded_e = err[:,0:self.seq_len].clone()

        src_mask = (padded_x != 0).unsqueeze(1).unsqueeze(2)

        batch_mask = torch.zeros(
            padded_x.shape, dtype=torch.bool, device=x.device
        )

        masking_fraction = int(self.frac * self.seq_len)
        # red mask
        if self.red:

            masking_fraction = self.seq_len - masking_fraction
            batch_mask[:, masking_fraction:self.seq_len] = True
        # blue mask
        else:

            batch_mask[:, 0:masking_fraction] = True

        # Apply mask per batch
        # --- apply mask to flux ---
        masked_x = padded_x.clone()          # one clone for the whole batch
        masked_x[batch_mask] = 0.0

        masked_e = padded_e.clone()
        masked_e[batch_mask] = 0.0

        # --- vectorised normalisation (no per-sample loop) ---
        # count non-zero per sample: (batch,)
        non_zero_mask = padded_x != 0.0 #masked_x #!= 0.0
        count = non_zero_mask.sum(dim=1, keepdim=True).clamp(min=1)  # avoid div/0

        # sum of squares of non-zero elements per sample
        flux_sq_sum = (masked_x ** 2 * non_zero_mask).sum(dim=1, keepdim=True)
        div = torch.sqrt(flux_sq_sum / count)                         # (batch, 1)

        # normalise — div broadcasts across seq_len dimension
        normalised_flux = padded_x / div
        normalised_mask = masked_x / div
        normalised_err = padded_e / div
        normalised_err_mask = masked_e / div
        # --- clip outliers vectorised ---
        normalised_flux = normalised_flux.clamp(-15, 15)
        normalised_mask = normalised_mask.clamp(-15, 15)
        normalised_err = normalised_err.clamp(0, 15)
        # --- src_mask vectorised ---
        src_mask = (padded_x != 0).unsqueeze(1).unsqueeze(2)

        return normalised_mask, normalised_flux,normalised_err,normalised_err_mask,src_mask,  padded_w, src_mask, batch_mask


class RandomMaskLines(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len

    def forward(self, x, w,e, min_fraction=0.1, max_fraction=0.2):
        # --- split input (no clone needed, views are fine) ---
        padded_x = x[:, :self.seq_len]
        padded_e = e[:, :self.seq_len]
        padded_w = w[:, :self.seq_len]


        # --- single scalar on GPU, no numpy ---
        mask_prob = torch.empty(1, device=x.device).uniform_(
            min_fraction, max_fraction
        ).item()

        # --- vectorised mask: entire batch in one kernel launch ---
        batch_mask = torch.rand_like(padded_x) < mask_prob  # (batch, seq_len)

        # --- apply mask to flux ---
        masked_x = padded_x.clone()          # one clone for the whole batch
        masked_x[batch_mask] = 0.0

        masked_e = padded_e.clone()          # one clone for the whole batch
        masked_e[batch_mask] = 0.0
        masked_ebr = padded_e.clone()          # one clone for the whole batch
        masked_ebr[~batch_mask] = 0.0
        # --- vectorised normalisation (no per-sample loop) ---
        # count non-zero per sample: (batch,)
        non_zero_mask = padded_x != 0.0#masked_x #!= 0.0
        count = non_zero_mask.sum(dim=1, keepdim=True).clamp(min=1)  # avoid div/0

        # sum of squares of non-zero elements per sample
        flux_sq_sum = (masked_x ** 2 * non_zero_mask).sum(dim=1, keepdim=True)
        div = torch.sqrt(flux_sq_sum / count)                         # (batch, 1)

        # normalise — div broadcasts across seq_len dimension
        normalised_flux = padded_x / div
        normalised_mask = masked_x / div

        # --- clip outliers vectorised ---
        normalised_flux = normalised_flux.clamp(-15, 15)
        normalised_mask = normalised_mask.clamp(-15, 15)

        normalised_e = masked_ebr / div
        normalised_emask = masked_e / div

        # --- clip outliers vectorised ---
        normalised_e = normalised_e.clamp(0, 15)
        normalised_emask = normalised_emask.clamp(0, 15)
        # --- src_mask vectorised ---
        src_mask = (padded_x != 0).unsqueeze(1).unsqueeze(2)

        return normalised_mask, normalised_flux, padded_w, src_mask, batch_mask, normalised_e, normalised_emask


class RandomMaskChunks(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len

    def forward(self, x, w, e,max_chunks=3, min_fraction=0.07, max_fraction=0.15):
        batch_size = x.shape[0]
        device = x.device

        # --- split input, no unnecessary clones ---
        padded_x = x[:, :self.seq_len]
        padded_e = e[:, :self.seq_len]
        padded_w = w[:, :self.seq_len]


        # --- compute chunk size bounds once ---
        min_chunk = int(min_fraction * self.seq_len)
        max_chunk = int(max_fraction * self.seq_len)

        # --- build batch_mask fully on GPU ---
        batch_mask = torch.zeros(
            batch_size, self.seq_len, dtype=torch.bool, device=device
        )

        # --- sample all random values at once on CPU (unavoidable for
        #     variable-length chunks), but minimise calls ---
        num_chunks_per_sample = torch.randint(
            1, max_chunks + 1, (batch_size,)
        )  # (batch,) — on CPU, one call

        for b in range(batch_size):
            n_chunks = num_chunks_per_sample[b].item()

            # sample all chunk starts and sizes in two calls
            starts = torch.randint(0, self.seq_len, (n_chunks,), device=device)
            sizes  = torch.randint(min_chunk, max_chunk + 1, (n_chunks,), device=device)

            # vectorise over chunks: offsets shape (n_chunks, max_size)
            max_size = sizes.max().item()
            offsets  = torch.arange(max_size, device=device).unsqueeze(0)  # (1, max_size)
            starts_  = starts.unsqueeze(1)                                  # (n_chunks, 1)
            sizes_   = sizes.unsqueeze(1)                                   # (n_chunks, 1)

            # indices with wrap-around: (n_chunks, max_size)
            indices = (starts_ + offsets) % self.seq_len

            # mask out padding beyond each chunk's actual size
            valid   = offsets < sizes_                                      # (n_chunks, max_size)
            indices = indices[valid]                                        # flat valid indices

            # single scatter write for all chunks in this sample
            batch_mask[b].scatter_(0, indices, True)

        # --- apply mask ---
        masked_x = padded_x.clone()
        masked_x[batch_mask] = 0.0
        masked_e = padded_e.clone()
        masked_e[batch_mask] = 0.0
        masked_ebr = padded_e.clone()
        masked_ebr[~batch_mask] = 0.0
        # --- vectorised normalisation (identical to RandomMaskLines) ---
        non_zero_mask = padded_x != 0.0 #masked_x #!= 0.0
        count    = non_zero_mask.sum(dim=1, keepdim=True).clamp(min=1)
        flux_sq  = (masked_x ** 2 * non_zero_mask).sum(dim=1, keepdim=True)
        div      = torch.sqrt(flux_sq / count)                             # (batch, 1)

        normalised_flux = (padded_x / div).clamp(-15, 15)
        normalised_mask = (masked_x  / div).clamp(-15, 15)
        normalised_e = (masked_ebr / div).clamp(0, 15)
        normalised_emask = (masked_e  / div).clamp(0, 15)
        src_mask  = (normalised_mask != 0).unsqueeze(1).unsqueeze(2)

        return normalised_mask, normalised_flux, padded_w, src_mask, batch_mask, normalised_e, normalised_emask

