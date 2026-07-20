import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import numpy.ma as ma
import copy
import time
from astropy.table import Table
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scheduled import CosineSchedule, WSDSchedule
from scipy.stats import chisquare
import h5py
from scipy.interpolate import interp1d
from functools import partial

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, device, probs = False):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        self.device = device
        self.probs = probs

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model).to(self.device, dtype=torch.float32) # Query transformation
        self.W_k = nn.Linear(d_model, d_model).to(self.device, dtype=torch.float32) # Key transformation
        self.W_v = nn.Linear(d_model, d_model).to(self.device, dtype=torch.float32) # Value transformation
        self.W_o = nn.Linear(d_model, d_model).to(self.device, dtype=torch.float32) # Output transformation


    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        Q = Q.to(self.device, dtype=torch.float32)
        K = K.to(self.device, dtype=torch.float32)
        V = V.to(self.device, dtype=torch.float32)


        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)       #[batch, heads, input, input]
        attn_scores.to(self.device)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -float("Inf"))    # attention = - inf if masked


        # Potential stabiliser
        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True).values


        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)   ##[batch, heads, input, input]

        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)    #[batch, heads, input, d_model/num_heads]

        if self.probs == False:
            return output

        else:
            return output , attn_probs


    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention

        batch_size, seq_length, self.d_model = x.size()
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)  #[batch, heads, input, d_model/num_heads]

        return x

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)   #[batch, input, d_model]

        return x

    def forward(self, Q, K, V, mask=None):

        # Apply linear transformations and split heads

        Q = self.split_heads(self.W_q(Q))   #[batch, heads, input, d_model/num_heads]
        K = self.split_heads(self.W_k(K))    #[batch, heads, input, d_model/num_heads]
        V = self.split_heads(self.W_v(V))    #[batch, heads, input, d_model/num_heads]
        # Perform scaled dot-product attention

        if self.probs == False:

            attn_output = self.scaled_dot_product_attention(Q, K, V, mask) #[batch, heads, input, d_model/num_heads]

        else:
            attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask) #[batch, heads, input, d_model/num_heads]

        attn_output = attn_output.to(self.device, torch.float32)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))   #[batch, input, d_model]

        if self.probs == False:
            return output
        else:
            return output, attn_probs

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, device):
        super(PositionWiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff, dtype=torch.float32).to(device)
        self.fc2 = nn.Linear(d_ff, d_model, dtype=torch.float32).to(device)
        self.relu = nn.ReLU()
        self.device = device

    def forward(self, x):

        x = x.to(self.device)

        x = self.fc2(self.relu(self.fc1(x)))   #[batch, input, d_model]

        x = x.to(self.device)

        return x



class WavelengthPositionalEncodingVaswaniEquivalent(nn.Module):
    """ This is equivalent to the original transformer positional encoding
    (Vaswani et al., 2017) when encoding the wavelength eqidistantly in units of choice (nn, AA even log(AA))
    """
    def __init__(self, d_model: int, min_w: float, max_w: float, no_wavelengths, device: str | torch.device = "cpu"):
        super().__init__()

        self.d_model = d_model
        self.device = device
        self.register_buffer("min_w", torch.tensor(min_w, dtype=torch.get_default_dtype(), device=device))

        dw = (max_w - min_w) / (no_wavelengths - 1)
        max_omega = 1 / dw
        min_omega = 1 / dw * 1e-4 ** ((d_model - 2)/d_model)

        omega = np.geomspace(max_omega, min_omega, d_model // 2)
        omega = torch.tensor(omega, dtype=torch.get_default_dtype(), device=device)

        self.register_buffer("omega", omega)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        w : Tensor[..., L]
        """
        x = w - self.min_w

        sin = torch.sin(x.unsqueeze(-1) * self.omega)
        cos = torch.cos(x.unsqueeze(-1) * self.omega)

        out = torch.empty((*sin.shape[:-1], self.d_model),
                          dtype=sin.dtype,
                          device=sin.device)
        out[..., ::2] = sin
        out[..., 1::2] = cos

        return out

class EncoderLayer(nn.Module):
    """Encoder Layer with pre-layer  normalisation"""
    def __init__(self, d_model, num_heads, d_ff, device, probs):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, device, probs)
        self.device = device
        self.probs = probs
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, device)
        self.norm1 = nn.LayerNorm(d_model).to(device, dtype=torch.float32)
        self.norm2 = nn.LayerNorm(d_model).to(device, dtype=torch.float32)

    def forward(self, x, mask):
        x = x.to(self.device)

        _x = self.norm1(x).to(self.device)

        if self.probs == False:
            attn_output = self.self_attn(_x, _x, _x, mask)

        else:
            attn_output, probs = self.self_attn(_x,_x,_x, mask)

        x = x + attn_output

        _x = self.norm2(x).to(self.device)

        ff_output = self.feed_forward(_x).to(self.device)   #[batch, input, d_model]

        x = x + ff_output

        if self.probs == False:
            return x
        else:
            return x, probs


class NonLinearEmbeddingLayer(nn.Module):
    def __init__(self, input_size, output_size,device, activation_fn): # took out output size
        super(NonLinearEmbeddingLayer, self).__init__()

        self.activation_fn = activation_fn

        self.weights = nn.Parameter(torch.empty(output_size, dtype=torch.float32))

        self.bias = nn.Parameter(torch.empty(output_size, dtype=torch.float32))

        self.device = device


        nn.init.xavier_uniform_(self.weights.unsqueeze(0))
        nn.init.xavier_uniform_(self.bias.unsqueeze(0))

    def forward(self, x):

        x = x.to(self.device, dtype=torch.float32)

        x_exp = x.unsqueeze(-1)

        x_exp = x_exp.to(self.device)
        w = self.weights
        w = w.to(self.device)
        lin = x_exp * w
        lin = lin.to(self.device)
        b = self.bias
        b = b.to(self.device)
        lin_b = lin + b
        lin_b = lin_b.to(self.device)
        out = self.activation_fn(lin_b)
        out = out.to(self.device)
        return out


class MultiChannelEmbeddingLayer(nn.Module):
    def __init__(self, n_inputs, output_size, device, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn
        self.linear = nn.Linear(n_inputs, output_size)
        self.device = device
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # x: (B, L, n_inputs) — e.g., [flux, flux_error] stacked on last dim
        x = x.to(self.device, dtype=torch.float32)
        return self.activation_fn(self.linear(x))


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length,device, wav_min, wav_max,num_registers,probs = False):
        super(Transformer, self).__init__()

        self.device = device
        self.probs = probs
        self.num_registers = num_registers

        self.encoder_embedding = MultiChannelEmbeddingLayer(2, d_model, device, torch.relu)


        self.mu_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.logvar_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        nn.init.zeros_(self.logvar_head[-1].bias)
        nn.init.normal_(self.logvar_head[-1].weight, std=0.01)


        self.positional_encoding = WavelengthPositionalEncodingVaswaniEquivalent(d_model,wav_min,wav_max, max_seq_length, device)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, device, probs) for _ in range(num_layers)])

        # Initialize register tokens (trainable)
        self.register_tokens = nn.Parameter(torch.randn(1, num_registers, d_model) * 0.01)#.to(device, dtype = torch.float32)

        #linear layer
        self.fc = nn.Linear(d_model, 1).to(device,dtype=torch.float32)

    def generate_mask(self, src):

        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        return src_mask

    def forward(self, src,errors, wavelength, training=True, varian = True,scalar = None):

        src = src.to(self.device, dtype=torch.float32)
        wavelength = wavelength.to(self.device, dtype=torch.float32)
        errors = errors.to(self.device, dtype = torch.float32)
        self.varian = varian

        src_mask = self.generate_mask(src).to(self.device)  # (batch, 1, 1, seq_len)

        if self.training:

            noise = torch.randn_like(src) * errors
            perturbed = src + noise
            inputs = torch.stack([perturbed, errors], dim = -1)


        else:


            errors = errors.to(self.device)

            inputs = torch.stack([src, errors], dim = -1)


        x = (self.encoder_embedding(inputs)).to(self.device)

        pos_enc = self.positional_encoding(wavelength).to(self.device)

        enc_output = x + pos_enc


        if self.num_registers > 0:


            regs = self.register_tokens.expand(src.size(0), -1, -1).to(enc_output.device)
            enc_output = torch.cat([enc_output, regs], dim=1)
            batch_size = enc_output.size(0)

            reg_key_mask = torch.ones(batch_size, 1, 1, self.num_registers,
                              device=enc_output.device, dtype=src_mask.dtype)
            src_mask = torch.cat([src_mask, reg_key_mask], dim=-1)

            if self.probs:

                all_attn_probs = []

                for enc_layer in self.encoder_layers:

                    enc_output, attn_probs = enc_layer(enc_output, src_mask)

                    all_attn_probs.append(attn_probs.detach().cpu())

                enc_output = enc_output[:, :-self.num_registers, :]


            else:
                for enc_layer in self.encoder_layers:

                    enc_output = enc_layer(enc_output, src_mask)



                enc_output = enc_output[:, :-self.num_registers, :]




        else:
            if self.probs:


                all_attn_probs = []
                for enc_layer in self.encoder_layers:

                    enc_output, probs = enc_layer(enc_output, src_mask)
                    all_attn_probs.append(probs.detach().cpu())


            else:
                for enc_layer in self.encoder_layers:

                    enc_output = enc_layer(enc_output, src_mask)


        if self.varian:

            mu = self.mu_head(enc_output)
            log_var = self.logvar_head(enc_output)  # (B, L)
            log_var = log_var.clamp(-10, 10)
            if self.probs:
                return mu, log_var, all_attn_probs
            else:
                return mu, log_var

        else:

            mu = self.mu_head(enc_output)

            if self.probs:
                return mu, all_attn_probs
            else:
                return mu


class NumericalDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self):
        return len(self.arrays[0])

    def __getitem__(self, idx):
        return tuple(arr[idx] for arr in self.arrays)
