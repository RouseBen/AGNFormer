import numpy as np

import torch
import torch.nn as nn

class NoMask(nn.Module):
    def __init__(self, seq_len = 2500, random_seed = 48):
        super().__init__()
        self.seed = random_seed
        self.seq_len = seq_len
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def forward(self, x,y, w):
        batch_size = x.shape[0]
        padded_x = x[:,0:self.seq_len].clone()
        padded_y = y[:,0:self.seq_len].clone()
        padded_w = w[:,0:self.seq_len].clone()
        padded_l = x[:,self.seq_len:].clone()
        #padded_l[:,2] = padded_l[:,2]/max_lum
        src_mask = (padded_x != 0).unsqueeze(1).unsqueeze(2)

        normalised_flux = torch.zeros_like(padded_x)
        normalised_mask = torch.zeros_like(padded_x)
        for b in range(batch_size):
             spec = padded_x[b]
             count = len(spec[spec != 0.0])

             fluxarr = torch.sum(spec[spec!=0.0]**2)

             div = torch.sqrt(fluxarr/count)

             normalised_preflux = padded_x[b]/div
             normalised_preflux[(normalised_preflux > 15) | (normalised_preflux < -15)] = 0
             normalised_flux[b] = normalised_preflux

        return padded_x, padded_y, padded_w, src_mask
        #return normalised_flux, padded_w, padded_l, src_mask

class FixedMask(nn.Module):
    def __init__(self, seq_len, random_seed = 48, line_centres, mask_width_kms = 10000):
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
            padded_e = err[:,0:2500].clone()


        src_mask = (padded_x != 0).unsqueeze(1).unsqueeze(2)
        batch_mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool, device=x.device)


        for low, high in self.mask_ranges:
            batch_mask |= (padded_w >= low) & (padded_w <= high)
        mask_ranges = [(np.log10(1351.023),np.log10(1601.1293333333333)),(np.log10(1844.4),np.log10(1971.6)),(np.log10(2705.8131000000003) ,np.log10(2892.4209)),(np.log10(4196.957333333334),np.log10(4486.402666666667)) ,(np.log10(4700.590666666667) ,np.log10(5024.769333333334)),(np.log10(6345.789),np.log10(6783.430333333333))]
        #mask_ranges = [(np.log10(1100),np.log10(1260)),(np.log10(1262.0123333333333),np.log10(1349.0476666666666)),(np.log10(1351.023),np.log10(1444.197)),(np.log10(1497.8306666666667),np.log10(1601.1293333333333)),(np.log10(1844.4),np.log10(1971.6)),(np.log10(2705.8131000000003) ,np.log10(2892.4209)),(np.log10(4196.957333333334),np.log10(4486.402666666667)) ,(np.log10(4700.590666666667) ,np.log10(5024.769333333334)),(np.log10(6345.789),np.log10(6783.430333333333))]
        """for b in range(batch_size):
            wav = padded_w[b]
            rest = wav.cpu().numpy()

            if z is not None:

                for i in range(len(rest)):
                    rest[i] = np.log10((10**rest[i])/(1+z[b]))

            for low, high in mask_ranges:
                mask = (rest >= low) & (rest<=high)
                batch_mask[b, mask] = True
                #mask = (wav >= low) & (wav<=high)
                #batch_mask[b, mask] = True
        """
        batch_mask = torch.zeros(
            padded_x.shape, dtype=torch.bool, device=x.device
        )

        for wav_min, wav_max in mask_ranges:
            batch_mask |= (padded_w >= wav_min) & (padded_w <= wav_max)


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

        return normalised_mask, normalised_flux, normalised_err, normalised_err_mask,normed_br_err_mask, padded_w, padded_l, src_mask, batch_mask

class FixedMaskShuffle(nn.Module):
    def __init__(self, seq_len=2500, random_seed=48, shuffle=True, shuffle_per_sample=False):
        super().__init__()
        self.seed = random_seed
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.shuffle_per_sample = shuffle_per_sample
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def forward(self, x, w, err=None, max_lum=None, z=None):
        batch_size = x.shape[0]
        padded_x = x[:, 0:2500].clone()
        padded_w = w[:, 0:2500].clone()
        padded_l = x[:, 2500:].clone()
        if err is not None:
            padded_e = err[:, 0:2500].clone()
        if max_lum is not None:
            padded_l[:, 2] = padded_l[:, 2] / max_lum
        if z is not None:
            z = z[:].cpu().numpy()

        mask_ranges = [(np.log10(1351.023), np.log10(1601.1293333333333)), (np.log10(1844.4), np.log10(1971.6)), (np.log10(2705.8131000000003), np.log10(2892.4209)), (np.log10(4200.590666666667), np.log10(4500.769333333334)),(np.log10(4700.590666666667), np.log10(5050.769333333334)), (np.log10(6345.789), np.log10(6783.430333333333))]

        batch_mask = torch.zeros(padded_x.shape, dtype=torch.bool, device=x.device)
        for wav_min, wav_max in mask_ranges:
            batch_mask |= (padded_w >= wav_min) & (padded_w <= wav_max)

        # --- apply mask to flux ---
        masked_x = padded_x.clone()
        masked_x[batch_mask] = 0.0
        if err is not None:
            masked_e = padded_e.clone()
            masked_e_broad = padded_e.clone()
            masked_e[batch_mask] = 0.0
            masked_e_broad[~batch_mask] = 0.0

        # --- vectorised normalisation ---
        non_zero_mask = padded_x != 0.0
        count = non_zero_mask.sum(dim=1, keepdim=True).clamp(min=1)
        flux_sq_sum = (masked_x ** 2 * non_zero_mask).sum(dim=1, keepdim=True)
        div = torch.sqrt(flux_sq_sum / count)

        normalised_flux = padded_x / div
        normalised_mask = masked_x / div
        normalised_err = padded_e / div
        normalised_err_mask = masked_e / div
        normed_br_err_mask = masked_e_broad / div

        # --- clip outliers ---
        normalised_flux = normalised_flux.clamp(-15, 15)
        normalised_mask = normalised_mask.clamp(-15, 15)
        normalised_err = normalised_err.clamp(0, 15)
        normalised_err_mask = normalised_err_mask.clamp(0, 15)
        normed_br_err_mask = normed_br_err_mask.clamp(0, 15)

        # --- shuffle the sequence dimension ---
        if self.shuffle:
            if self.shuffle_per_sample:
                # independent permutation per sample
                perm = torch.argsort(torch.rand(batch_size, self.seq_len, device=x.device), dim=1)
            else:
                # one fixed permutation reused across the batch (deterministic given seed)
                g = torch.Generator(device=x.device)
                g.manual_seed(self.seed)
                perm = torch.randperm(self.seq_len, generator=g, device=x.device)
                perm = perm.unsqueeze(0).expand(batch_size, -1)

            def shuffle_seq(t):
                # t: (batch, seq_len)
                return torch.gather(t, 1, perm)

            normalised_flux = shuffle_seq(normalised_flux)
            normalised_mask = shuffle_seq(normalised_mask)
            normalised_err = shuffle_seq(normalised_err)
            normalised_err_mask = shuffle_seq(normalised_err_mask)
            normed_br_err_mask = shuffle_seq(normed_br_err_mask)
            padded_w = shuffle_seq(padded_w)
            batch_mask = shuffle_seq(batch_mask)

        # --- src_mask (computed after shuffle on shuffled flux) ---
        src_mask = (normalised_flux != 0).unsqueeze(1).unsqueeze(2)

        return normalised_mask, normalised_flux, normalised_err, normalised_err_mask, normed_br_err_mask, padded_w, padded_l, src_mask, batch_mask

class HalfMask(nn.Module):
    def __init__(self, seq_len = 2500, random_seed = 48):
        super().__init__()
        self.seed = random_seed
        self.seq_len = seq_len
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def forward(self, x, w,err):
        batch_size = x.shape[0]
        padded_x = x[:,0:2500].clone()
        padded_w = w[:,0:2500].clone()
        padded_l = x[:,2500:].clone()
        padded_e = err[:,0:2500].clone()
        #padded_l[:,2] = padded_l[:,2]/max_lum
        src_mask = (padded_x != 0).unsqueeze(1).unsqueeze(2)

        batch_mask = torch.zeros(
            padded_x.shape, dtype=torch.bool, device=x.device
        )
        # red mask
        #batch_mask[:, 1250:2500] = True
        # blue mask
        #batch_mask[:, 0:1250] = True
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

        return normalised_mask, normalised_flux,normalised_err,normalised_err_mask,src_mask,  padded_w, padded_l, src_mask, batch_mask

class PhotoMask(nn.Module):
    def __init__(self, seq_len=2500, random_seed=48, photo_spacing_angstrom=100.0):
        super().__init__()
        self.seed = random_seed
        self.seq_len = seq_len
        self.photo_spacing_angstrom = photo_spacing_angstrom
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def forward(self, x, w, err):
        batch_size = x.shape[0]
        padded_x = x[:, 0:2500].clone()
        padded_w = w[:, 0:2500].clone()
        padded_l = x[:, 2500:].clone()
        padded_e = err[:, 0:2500].clone()

        src_mask = (padded_x != 0).unsqueeze(1).unsqueeze(2)

        batch_mask = torch.zeros(
            padded_x.shape, dtype=torch.bool, device=x.device
        )
        # blue mask
        batch_mask[:, 0:1250] = True

        # --- photometry-like unmasking every 100Å ---
        # padded_w shape: (batch, 2500); wavelengths may vary per spectrum,
        # so we compute the photometry indices per sample.
        # For each sample, find indices where wavelength % 100 is closest to 0.
        w_np = padded_w.detach().cpu().numpy()  # (batch, 2500)

        photo_mask = torch.zeros_like(batch_mask)  # True = unmask this point

        for b in range(batch_size):
            wl = w_np[b]  # (2500,)
            valid = wl > 0  # ignore padding

            # Find the nearest index to each 100Å grid point within the masked region
            if valid.any():
                w_min = wl[valid].min()
                w_max = wl[valid].max()
                grid_points = np.arange(
                    np.ceil(w_min / self.photo_spacing_angstrom) * self.photo_spacing_angstrom,
                    w_max,
                    self.photo_spacing_angstrom
                )
                for target_w in grid_points:
                    # Only consider indices that are masked (in the blue half)
                    candidates = np.where(valid & (wl <= wl[1249]))[0]
                    if len(candidates) == 0:
                        continue
                    nearest = candidates[np.argmin(np.abs(wl[candidates] - target_w))]
                    photo_mask[b, nearest] = True

        # Unmask the photometry points
        batch_mask = batch_mask & ~photo_mask

        # --- apply mask to flux ---
        masked_x = padded_x.clone()
        masked_x[batch_mask] = 0.0
        masked_e = padded_e.clone()
        masked_e[batch_mask] = 0.0

        # --- vectorised normalisation ---
        non_zero_mask = padded_x != 0.0
        count = non_zero_mask.sum(dim=1, keepdim=True).clamp(min=1)
        flux_sq_sum = (masked_x ** 2 * non_zero_mask).sum(dim=1, keepdim=True)
        div = torch.sqrt(flux_sq_sum / count)

        normalised_flux = padded_x / div
        normalised_mask = masked_x / div
        normalised_err = padded_e / div
        normalised_err_mask = masked_e / div

        normalised_flux = normalised_flux.clamp(-15, 15)
        normalised_mask = normalised_mask.clamp(-15, 15)
        normalised_err = normalised_err.clamp(0, 15)

        src_mask = (padded_x != 0).unsqueeze(1).unsqueeze(2)

        return normalised_mask, normalised_flux, normalised_err, normalised_err_mask, src_mask, padded_w, padded_l, src_mask, batch_mask


class RandomMask(nn.Module):
    def __init__(self, seq_len = 2500, random_seed = 48):
        super().__init__()
        #self.seed = random_seed
        self.seq_len = seq_len
        #torch.manual_seed(random_seed)
        #np.random.seed(random_seed)

    def forward(self, x, w,err=None, denominators=[2,4,6,8],std_fraction=0.2,
            min_fraction=0.15,max_fraction=0.4):
        batch_size = x.shape[0]
        padded_x = x[:,0:self.seq_len].clone()
        padded_w = w[:,0:self.seq_len].clone()
        padded_l = x[:,self.seq_len:].clone()
        if err is not None:
            padded_e = err[:,0:self.seq_len].clone()
        ## Zero padding
        #if x.shape[1] < self.seq_len:

        #    pad_amount = self.seq_len - x.shape[1]
        #    padded_x = F.pad(x, (0, pad_amount), mode="constant", value=0)
        #    padded_w = F.pad(w, (0, pad_amount), mode="constant", value=0)
        #    src_mask = (padded_x != 0).unsqueeze(1).unsqueeze(2)
        #else:
        #    padded_x = x[:,0:4500]
        #    padded_w = w[:,0:4500]
        src_mask = (padded_x != 0).unsqueeze(1).unsqueeze(2)
        # random masking
        # choose random number for denom
        # random for num
        # Randomly initialise starting point and allow for wrap around
        # Choose denominator
        # Initialize mask tensor: [batch_size, seq_len]
        batch_mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool, device=x.device)

        for b in range(batch_size):

            denom = np.random.choice(denominators)
            # Chunk size
            #chunk_size = self.seq_len // denom
            chunk_size = np.random.randint(int(min_fraction * self.seq_len), int(max_fraction * self.seq_len) + 1)
            # Draw numerator from Gaussian
            mean = denom / 2
            std = denom * std_fraction
            #print(mean, std)

            numerator = int(np.round(np.random.normal(mean, std)))
            #print(numerator)
            # Clip numerator to valid range
            numerator = np.clip(numerator, 1, denom)
            #print(numerator)
            # Compute mask fraction and adjust if out of range
            frac = numerator / denom
            #print(frac)
            if frac < min_fraction:
                numerator = int(np.ceil(min_fraction * denom))
                #print(numerator)
            elif frac > max_fraction:
                numerator = int(np.floor(max_fraction * denom))
                #print(numerator)

            #qeowfubqof
            # Random start position for 0th chunk
            start = np.random.randint(0, self.seq_len)
            # Mask the chunk with wrap-around
            for i in range(chunk_size):
                batch_mask[b, (start + i) % self.seq_len] = True
            ## Indices for all chunks
            #chunk_starts = [(start + chunk_size)]# % self.seq_len for i in range(denom)]

            # Randomly select numerator chunks to mask
            #mask_chunks = np.random.choice(denom, numerator, replace=False)
            ## Apply masking with wrap-around
            #for idx in mask_chunks:
            #   # s = chunk_starts[idx]
            #    #for i in range(chunk_size):
            #        batch_mask[b, (s + i) % self.seq_len] = True


        # Apply mask per batch
        masked_spec = padded_x.clone()
        #masked_err = padded_e.clone()
        #masked_wav = padded_w.clone()
        masked_spec[batch_mask] = 0  # or another value if you want
        if err is not None:
            masked_err = padded_e.clone()
            masked_err[batch_mask] = 0
        # normalising

        normalised_flux = torch.zeros_like(padded_x)
        normalised_mask = torch.zeros_like(padded_x)
        for b in range(batch_size):
             spec = masked_spec[b]
             count = len(spec[spec != 0.0])

             fluxarr = torch.sum(spec[spec!=0.0]**2)

             div = torch.sqrt(fluxarr/count)

             normalised_preflux = padded_x[b]/div
             normalised_preflux[(normalised_preflux > 15) | (normalised_preflux < -15)] = 0
             normalised_flux[b] = normalised_preflux

             normalised_premask = masked_spec[b]/div
             normalised_premask[(normalised_premask > 15) | (normalised_premask < -15)] = 0
             normalised_mask[b] = normalised_premask

        if err is not None:
            return normalised_mask, normalised_flux, padded_w, padded_l, src_mask, batch_mask, masked_err
        else:
            return normalised_mask, normalised_flux, padded_w, padded_l, src_mask, batch_mask



class RandomMaskLines(nn.Module):
    def __init__(self, seq_len=2500):
        super().__init__()
        self.seq_len = seq_len

    def forward(self, x, w,e, min_fraction=0.1, max_fraction=0.2):
        # --- split input (no clone needed, views are fine) ---
        padded_x = x[:, :self.seq_len]
        padded_e = e[:, :self.seq_len]
        padded_w = w[:, :self.seq_len]
        padded_l = x[:, self.seq_len:]

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

        return normalised_mask, normalised_flux, padded_w, padded_l, src_mask, batch_mask, normalised_e, normalised_emask

"""
class RandomMaskLines(nn.Module):
    def __init__(self, seq_len = 2500, random_seed = 48):
        super().__init__()
        #self.seed = random_seed
        self.seq_len = seq_len
        #torch.manual_seed(random_seed)
        #np.random.seed(random_seed)

    def forward(self, x, w, min_fraction=0.1,max_fraction=0.2):
        batch_size = x.shape[0]
        padded_x = x[:,0:self.seq_len].clone()
        padded_w = w[:,0:self.seq_len].clone()
        padded_l = x[:,self.seq_len:].clone()

        self.min_fraction = min_fraction
        self.max_fraction = max_fraction

        mask_prob = np.random.uniform(self.min_fraction, self.max_fraction)


        src_mask = (padded_x != 0).unsqueeze(1).unsqueeze(2)

        batch_mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool, device=x.device)

        normalised_flux = torch.zeros_like(padded_x)
        normalised_mask = torch.zeros_like(padded_x)

        for b in range(batch_size):



            mask = torch.rand_like(padded_x[b]) < mask_prob
            masked = padded_x[b].clone()
            masked[mask] = 0.0   # or some special value

            batch_mask[b][mask] = True

            count = len(masked[masked != 0.0])

            fluxarr = torch.sum(masked[masked!=0.0]**2)

            div = torch.sqrt(fluxarr/count)

            normalised_preflux = padded_x[b]/div
            normalised_preflux[(normalised_preflux > 15) | (normalised_preflux < -15)] = 0
            normalised_flux[b] = normalised_preflux

            normalised_premask = masked/div
            normalised_premask[(normalised_premask > 15) | (normalised_premask < -15)] = 0
            normalised_mask[b] = normalised_premask

        return normalised_mask, normalised_flux, padded_w, padded_l, src_mask, batch_mask
"""
class RandomMaskChunks(nn.Module):
    def __init__(self, seq_len=2500):
        super().__init__()
        self.seq_len = seq_len

    def forward(self, x, w, e,max_chunks=3, min_fraction=0.07, max_fraction=0.15):
        batch_size = x.shape[0]
        device = x.device

        # --- split input, no unnecessary clones ---
        padded_x = x[:, :self.seq_len]
        padded_e = e[:, :self.seq_len]
        padded_w = w[:, :self.seq_len]
        padded_l = x[:, self.seq_len:]

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

        return normalised_mask, normalised_flux, padded_w, padded_l, src_mask, batch_mask, normalised_e, normalised_emask

"""
class RandomMaskChunks(nn.Module):
    def __init__(self, seq_len = 2500, random_seed = 48):
        super().__init__()
        #self.seed = random_seed
        self.seq_len = seq_len
        #torch.manual_seed(random_seed)
        #np.random.seed(random_seed)

    def forward(self, x, w,max_chunks=3,min_fraction=0.07,max_fraction=0.15):
        batch_size = x.shape[0]
        padded_x = x[:,0:self.seq_len].clone()
        padded_w = w[:,0:self.seq_len].clone()
        padded_l = x[:,self.seq_len:].clone()

        src_mask = (padded_x != 0).unsqueeze(1).unsqueeze(2)

        batch_mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool, device=x.device)

        for b in range(batch_size):

            chunks = np.random.randint(max_chunks) + 1
            # Chunk size
            #chunk_size = self.seq_len // denom
            for c in range(chunks):
                chunk_size = np.random.randint(int(min_fraction * self.seq_len), int(max_fraction * self.seq_len) + 1)

                start = np.random.randint(0, self.seq_len)
                # Mask the chunk with wrap-around
                for i in range(chunk_size):
                    batch_mask[b, (start + i) % self.seq_len] = True
            ## Indices for all chunks
            #chunk_starts = [(start + chunk_size)]# % self.seq_len for i in range(denom)]

            # Randomly select numerator chunks to mask
            #mask_chunks = np.random.choice(denom, numerator, replace=False)
            ## Apply masking with wrap-around
            #for idx in mask_chunks:
            #   # s = chunk_starts[idx]
            #    #for i in range(chunk_size):
            #        batch_mask[b, (s + i) % self.seq_len] = True


        # Apply mask per batch
        masked_spec = padded_x.clone()
        #masked_err = padded_e.clone()
        #masked_wav = padded_w.clone()
        masked_spec[batch_mask] = 0  # or another value if you want

        # normalising

        normalised_flux = torch.zeros_like(padded_x)
        normalised_mask = torch.zeros_like(padded_x)
        for b in range(batch_size):
             spec = masked_spec[b]
             #count = len(spec[spec != 0.0])
             count = len(spec)
             #fluxarr = torch.sum(spec[spec!=0.0]**2)
             fluxarr = torch.sum(spec**2)
             div = torch.sqrt(fluxarr/count)

             normalised_preflux = padded_x[b]/div
             normalised_preflux[(normalised_preflux > 15) | (normalised_preflux < -15)] = 0
             normalised_flux[b] = normalised_preflux

             normalised_premask = masked_spec[b]/div
             normalised_premask[(normalised_premask > 15) | (normalised_premask < -15)] = 0
             normalised_mask[b] = normalised_premask


        src_mask = (normalised_mask != 0).unsqueeze(1).unsqueeze(2)

        return normalised_mask, normalised_flux, padded_w, padded_l, src_mask, batch_mask


"""
