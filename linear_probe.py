import h5py
import numpy as np 
import matplotlib.pyplot as plt 


fig, axs = plt.subplots(1,8, figsize = (12,6), sharey = True, sharex = True)
fig2, axs2 = plt.subplots(1,8, figsize = (12,8), sharey = True, sharex = True)
#fig, axs = plt.subplots(1,1,figsize = (10, 8))
label_plotted = False
for i in range(8):
    #fig, axs = plt.subplots(3,2, figsize = (12,8), sharey = 'row', sharex = False)

    with h5py.File(f"/scratch/dg97/br1378/data/layer_outputs/layer_{i}_data.h5", "r") as hf:
        #Load the datasets

        #flux = np.array(hf['flux'])
        #print(flux)
        #osdvoiwnv
        mass = np.array(hf['mass'])
        fe = np.array(hf['fe_uv_ew'])
        lum = np.array(hf['lum'])
        # Create a mask for sources with mass <= 20000
        mask = (np.isfinite(np.log10(fe))) & (lum > 40)
        hidden_states = np.array(hf["hidden_states"])[mask]
        lum = np.array(hf['lum'])[mask]  # best with z score
        #lum = (lum - np.mean(lum))/(np.std(lum))
        lum = (lum - np.min(lum)) / (np.max(lum) - np.min(lum))
        mass = np.array(hf['mg_fwhm'])[mask] # best with min max
        mass = mass / np.max(mass)
        #mass = (mass - np.min(mass)) / (np.max(mass) - np.min(mass))
        ew = np.array(hf['ew'])[mask] # best with min max
        ew = np.log10(ew)
        ew = (ew - np.min(ew)) / (np.max(ew) - np.min(ew))
        #ew = np.log10(ew)
        #bal = np.array(hf['bal_prob'])[mask]
        bi = np.array(hf['cont'])[mask] #minmax
        bi = (bi - np.min(bi)) / (np.max(bi) - np.min(bi))
        blue = np.array(hf['civ_blue'])[mask] # best with /max
        blue = (blue - np.min(blue))/(np.max(blue) -np.min(blue))
        fe = np.array(hf['fe_uv_ew'])[mask] # minmax
        fe = np.log10(fe)
        fe = (fe - np.min(fe)) / (np.max(fe) - np.min(fe))
        he = np.array(hf['he_fwhm'])[mask]
        he = (he-np.min(he)) / (np.max(he) - np.min(he))
        l2500 = np.array(hf['lum_2500'])[mask]
        l2500 = (l2500-np.min(l2500)) / (np.max(l2500) - np.min(l2500))
        #fe = np.log10(fe)


    #mass = mass/np.max(mass)
    #blue = blue/np.max(blue)
    #x = lum #np.log10(fe)
    #x_scaled = (x - np.mean(x)) / np.std(x)
    #x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
    #x_scaled = x/np.max(x)
    #x_scaled = x
    # input FWHM, check if it needs to be noramlised
    x_scaled = mass

    X_train, X_test, y_train, y_test = train_test_split(hidden_states, x_scaled, test_size = 0.3, random_state = 1352, shuffle = False)

    X_test, X_val,y_test,y_val = train_test_split(X_test, y_test, test_size = 0.1, random_state = 1352, shuffle = False)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)  # [N, 1]

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)  # [N, 1]

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)  # [N, 1]

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    model = ParamRegressor(hidden_dim=X_train.shape[-1])
    model = model.to("cuda")
    #lrs = [1e-5,3e-5,1e-4,3e-4, 1e-3,3e-3]
    #lrs = [1e-5,3e-5]
    lrs = [1e-4]
    warmup_steps = 300#5000
    total_steps  = len(train_loader) * 50# 360 # 50040

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + maths.cos(maths.pi * progress))



    for l in range(len(lrs)):
        base_lr = lrs[l]
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)#, weight_decay = 0.01)#1e-4)
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        criterion = nn.MSELoss()
        loss_array = []
        test_loss_array = []

        global_step = 0

        for epoch in range(50):#360):
            model.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                pred, _ = model(batch_x)
                loss = criterion(pred, batch_y)
                #l1_loss = F.l1_loss(pred,batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #scheduler.step()
                global_step += 1
                total_loss += loss.item()
                loss_array.append(loss.item())
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                    pred, _  = model(batch_x)
                    testloss = criterion(pred, batch_y)
                    test_loss += testloss.item()
                    test_loss_array.append(testloss.item())
        #line, = axs[0,i].plot(loss_array, '-', alpha=0.2)#, label=f"layer {i}")

        #x = np.linspace(0,len(loss_array),len(test_loss_array))
        #colour = line.get_color()
        #Use the same colour in another line
        #axs[0,i].plot(x, test_loss_array, '--',color = colour)#, label = f"lr= {base_lr}")
        #axs[0,i].set_yscale('log')
        test_lum = []
        test_pred = []
        all_attn_weights = []
        model.eval()
        mse_list = []
        mae_list = []
        test_lum = []
        test_pred = []
        all_attn_weights = []
        with torch.no_grad():
            l1 = 0
            mse = 0
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                pred, attn_weights = model(batch_x)

                l1_loss = F.l1_loss(pred, batch_y, reduction='none')
                mse_loss = F.mse_loss(pred, batch_y, reduction='none')

                # Flatten and move to CPU
                l1_values = l1_loss.view(-1).cpu().numpy()
                mse_values = mse_loss.view(-1).cpu().numpy()

                mae_list.extend(l1_values)
                mse_list.extend(mse_values)

                test_lum.extend(batch_y.cpu().detach().numpy())
                test_pred.extend(pred.cpu().detach().numpy())

                #for a in range(len(attn_weights[:, 0])):
                    #axs2[i].plot(wavelength, attn_weights[a, :].squeeze().cpu().numpy(), '-', color='gray', alpha=0.1)
                    #all_attn_weights.append(attn_weights[a, :].cpu().numpy())

        # Plotting
                axs[i].plot(batch_y.cpu().detach().numpy(), pred.cpu().detach().numpy(), 'ko')
                if not label_plotted:
                    axs[i].plot(batch_y.cpu().detach().numpy(), batch_y.cpu().detach().numpy(), 'r-', label = '1:1')
                    label_plotted = True
                else:
                    axs[i].plot(batch_y.cpu().detach().numpy(), batch_y.cpu().detach().numpy(), 'r-')

        axs[i].set_box_aspect(1)
        axs[i].set_xlabel('True')
        axs[0].set_ylabel('Predicted')
        axs[0].legend()
        axs[i].xaxis.set_minor_locator(AutoMinorLocator())
        axs[i].yaxis.set_minor_locator(AutoMinorLocator())
        # Compute mean and std
        mse_mean = np.mean(mse_list)
        mse_std = np.std(mse_list)
        mae_mean = np.mean(mae_list)
        mae_std = np.std(mae_list)
        test_lum = np.array(test_lum)
        test_pred = np.array(test_pred)
        mean_true = np.mean(test_lum)
        ssres = 0
        sstot = 0
        for r in range(len(test_lum)):
            ssres += (test_lum[r] - test_pred[r])**2
            sstot += (test_lum[r] - mean_true) **2
        r_square = 1 - (ssres/sstot)
        print(f"nolum. Param = mass , layer = {i}, mse = {np.log10(mse_mean):.3f} ± {np.log10(1 + mse_std/mse_mean):.3f}, mae = {np.log10(mae_mean):.3f} ± {np.log10(1 + mae_std/mae_mean):.3f}")
        print(f"r squared = {r_square}")
        #all_attn_weights = np.stack(all_attn_weights, axis = 0)
        """
        mean_weights = np.mean(all_attn_weights, axis=0)
        sem_weights = np.std(all_attn_weights, axis=0) / np.sqrt(all_attn_weights.shape[0])

        median_weights = np.median(all_attn_weights, axis=0)
        axs2[i].plot(wavelength, median_weights.squeeze(), 'r-')
        iqr = np.subtract(*np.percentile(all_attn_weights, [75,25], axis=0))

        # --- Optional: save data ---
        np.savez(f"nolum_lum_attn_weights_{i}.npz",
            mean=mean_weights,
            sem=sem_weights,
            median=median_weights,
            iqr=iqr)#plt.savefig(f"edd_max_norm")
        """
fig.tight_layout()
fig.savefig('residuals_nolum_mass.png')
#fig2.savefig('attention_nolum_l1350.png')
owidnfwodinf