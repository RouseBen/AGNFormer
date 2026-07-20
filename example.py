import torch
import csv
from base import Transformer, NumericalDataset
from functions import NLLloss
from specmanip import FixedMask, RandomMaskLines, RandomMaskChunks, FractionalMask
from synthetic_spec import generate_synthetic_spectra
import os
import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scheduled import CosineSchedule, WSDSchedule
from scipy.stats import chisquare
from scipy.interpolate import interp1d
from functools import partial
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import math



# Create 1000 synthetic spectra in order to test
# Herre you would import your spectra
X, wav, err = generate_synthetic_spectra(n_spectra=1000, npix=2500)

wav_min = min(np.min(w) for w in wav if len(w) > 0)
wav_max = max(np.max(w) for w in wav if len(w) > 0)


X_train, X_test, wav_train, wav_test, err_train, err_test = train_test_split(X, wav, err, test_size = 0.3, random_state = 1352, shuffle = False)#True)

X_test, X_val, wav_test, wav_val, err_test, err_val = train_test_split(X_test, wav_test, err_test, test_size = 0.1, random_state = 1352, shuffle = False)#True)

train_dataset = NumericalDataset(torch.tensor(X_train, dtype=torch.float32),  torch.tensor(wav_train, dtype = torch.float32), torch.as_tensor(err_train, dtype = torch.float32))
test_dataset = NumericalDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(wav_test, dtype = torch.float32), torch.as_tensor(err_test, dtype = torch.float32))
val_dataset = NumericalDataset(torch.tensor(X_val, dtype = torch.float32), torch.tensor(wav_val, dtype = torch.float32), torch.as_tensor(err_val, dtype = torch.float32))

batch_size = 10


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle =True)
val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)

config = {"seq_len": len(X[0]),
          "d_model": 256,
          "n_heads":4,
          "n_layers":8,
          "d_ff":1024,
          "n_registers":1,
          "attn_probabilities":False,
          "epochs":12,
          "learning_rate":1e-4}



#device = torch.device("mps")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transformer = Transformer(config["seq_len"],config["seq_len"], config["d_model"],
                          config["n_heads"], config["n_layers"],
                          config["d_ff"], config["seq_len"],device,
                          wav_min, wav_max,config["n_registers"])

pretrain = True
weights = "transformer_weights.pt"
if pretrain:
  pass
else:
    transformer.load_state_dict(torch.load(f"{weights}", map_location=device))
transformer.to(device)


steps = len(train_loader) * config['epochs']

warmup_kwargs = {"steps": int((steps)/10)}

def run_validation():

    transformer.eval()
    val_loss = 0.0
    val_mse_loss = 0.0
    val_step_losses = []
    masking = FixedMask()

    with torch.no_grad():

        for val_xf, val_wav, err in test_loader:
            bx, by,_,be,bem,bw, sm, bm = masking(val_xf.to(device), val_wav.to(device), err=err.to(device))

            val_x = bx.to(device)
            val_y = by.to(device)

            be = be.to(device)
            val_wav = bw.to(device)

            src_mask = sm.to(device)
            val_mask = bm.to(device)

            bem = bem.to(device)

            val_out, log_var = transformer(val_x, val_wav, be,training = False, varian = True)
            val_out = val_out.squeeze(-1)

            vloss, vmseloss = loss(val_out, val_y, val_mask, flux_errors = bem, log_var = log_var)

            val_loss += vloss
            val_mse_loss += vmseloss

            val_step_losses.append(val_loss)

    transformer.train()
    return val_loss / len(test_loader), val_step_losses, val_mse_loss / len(test_loader)


training_losses = []
test_losses = []
train_step_losses_all = []
test_step_losses_all = []
globalstep = 0
scaler = GradScaler(device)


optimizer = optim.AdamW(transformer.parameters(), lr=config['learning_rate'], weight_decay = 0.05)
scheduler = WSDSchedule(final_lr=0.0, steps=steps+1, cooldown_len=0.2, base_lr=config['learning_rate'], warmup_kwargs=warmup_kwargs)

count = 0
training_losses = []
test_losses = []
train_step_losses_all = []
test_step_losses_all = []
globalstep = 0
scaler = GradScaler(device)
loss = NLLloss()

for epoch in range(config['epochs']):
    print('epoch=', epoch)
    transformer.train()
    train_loss = 0
    mse_train_loss = 0
    train_step_losses = []

    full_test_losses = []
    batch_len = 0
    for _, (bx,bw, be) in enumerate(train_loader):



        if (batch_len >= 0) and (batch_len < int(0.035*len(X_train))):
            masking = RandomMaskLines(seq_len=len(bx[0]))
            bx, by, bw, sm, bm, bem, bex = masking(bx,bw,be)


        elif (batch_len >= int(0.035*len(X_train))) and (batch_len < int(0.07*len(X_train))):
            masking = RandomMaskChunks(seq_len=len(bx[0]))
            bx, by, bw, sm, bm, bem,bex = masking(bx,bw,be)
            plt.figure()
            plt.plot(bw[0].cpu(), bx[0].cpu(), 'k-')
            plt.plot(bw[0].cpu(), by[0].cpu(), 'r-')
            plt.show()

        else:
            masking = FixedMask(seq_len=len(bx[0]))
            bx, by, be, bex,bem, bw, sm, bm = masking(bx,bw, be)

        if globalstep < len(scheduler.schedule):
            lr = scheduler.schedule[globalstep]
        else:
            lr = scheduler.schedule[-1]

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()
        globalstep += 1
        batch_len += 1

        with autocast("cuda:0"):

            batch_x = bx.to(device)
            batch_y = by.to(device)
            batch_e = bex.to(device)

            src_mask = sm.to(device)
            mask    = bm.to(device)

            wav     = bw.to(device)
            broad_e = bem.to(device)

            mu, log_var = transformer(batch_x,batch_e, wav,training=True, varian = True)

            targets = batch_y
            flux_errors = broad_e

            loss, mseloss = loss(mu, batch_y, mask, flux_errors=flux_errors, log_var = log_var)

            scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        train_step_losses_all.append(loss.item())
        mse_train_loss += mseloss.item()

    epochtrain = train_loss/len(train_loader)
    training_losses.append(epochtrain)
    mseepochtrain = mse_train_loss / len(train_loader)
    avg_test_loss, test_step_losses, val_mse_loss = run_validation()
    full_test_losses.append(test_step_losses)
    test_step_losses_all.append(test_step_losses)


    print(f"End of Epoch {epoch} Training loss = {epochtrain:.4f}, mse train = {mseepochtrain:.4f}, Validation Loss = {avg_test_loss:.4f}, mse val = {val_mse_loss:.4f}")





