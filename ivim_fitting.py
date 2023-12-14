import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import os
import pickle
import Chopper
import get_contour_masks
from numpy import diff
from utils import *
from utils import plot_signal_curve
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline, interp1d
import statsmodels.api as sm
from statsmodels.nonparametric import kernel_regression
from scipy.interpolate import UnivariateSpline
#import pyqt_fit.nonparam_regression as smooth

def biexp(b, Dp, D, f):
    return f*np.exp(-b*Dp) + (1-f)*np.exp(-b*D)
def triexp(b, D, Dp1, f1, Dp2, f2):
    return f1*np.exp(-b*Dp1) + f2*np.exp(-b*Dp2) + (1-f1-f2)*np.exp(-b*D)

class Net_ADC(nn.Module):
    def __init__(self, b_values):
        super(Net_ADC, self).__init__()
        self.bounds = [[1e-4*0.7, 4e-3*1.3], [0.6*0.7, 1.4*1.3]]
        self.b_values = b_values
        self.fc_layers = nn.ModuleList()
        for i in range(2): # 3 fully connected hidden layers
            self.fc_layers.extend([nn.Linear(len(b_values), len(b_values)),nn.BatchNorm1d(len(b_values)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values), 2)) #predict ADC and S(0)

    def forward(self, X):
        params = self.encoder(X) # Dp, Dt, Fp
        if torch.numel(torch.tensor(params.shape)) > 1:
            ADC = self.bounds[0][0] + torch.sigmoid(params[:, 0].unsqueeze(1)) * (self.bounds[0][1]-self.bounds[0][0])
            S0 = self.bounds[1][0] + torch.sigmoid(params[:,1].unsqueeze(1))* (self.bounds[1][1]-self.bounds[1][0])
        else:
            ADC = self.bounds[0][0] + torch.sigmoid(params[0]) * (self.bounds[0][1]-self.bounds[0][0])
            S0 = self.bounds[1][0] + torch.sigmoid(params[1].unsqueeze(1))* (self.bounds[1][1]-self.bounds[1][0])
        X_pred = (torch.exp(-self.b_values*ADC))

        return X_pred, ADC, S0

class Net_biexp(nn.Module):
    def __init__(self, b_values):
        super(Net_biexp, self).__init__()
        self.bounds = [[2e-3,0.1], [1e-4, 2e-3], [0, 0.4], [0.6, 1.4]]
        #extend ranges to allow sigmoid to stretch over
        for idx in range(len(self.bounds)):
            self.bounds[idx][0] *= 0.7
            self.bounds[idx][1] *= 1.3
        self.b_values = b_values
        self.fc_layers = nn.ModuleList()
        for i in range(2): # 3 fully connected hidden layers
            self.fc_layers.extend([nn.Linear(len(b_values), len(b_values)),nn.BatchNorm1d(len(b_values)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values), 4))

    def forward(self, X):
        params = self.encoder(X) # Dp, Dt, Fp
        if torch.numel(torch.tensor(params.shape)) > 1:
            Dp = self.bounds[0][0] + torch.sigmoid(params[:, 0].unsqueeze(1)) * (self.bounds[0][1]-self.bounds[0][0])
            D = self.bounds[1][0] + torch.sigmoid(params[:, 1].unsqueeze(1)) * (self.bounds[1][1]-self.bounds[1][0])        
            f = self.bounds[2][0] + torch.sigmoid(params[:, 2].unsqueeze(1))* (self.bounds[2][1]-self.bounds[2][0])
            S0 = self.bounds[3][0] + torch.sigmoid(params[:, 3].unsqueeze(1))*(self.bounds[3][1]-self.bounds[3][0])
        else:
            Dp = self.bounds[0][0] + torch.sigmoid(params[0]) * (self.bounds[0][1]-self.bounds[0][0])
            D = self.bounds[1][0] + torch.sigmoid(params[1]) * (self.bounds[1][1]-self.bounds[1][0])        
            f = self.bounds[2][0] + torch.sigmoid(params[2])* (self.bounds[2][1]-self.bounds[2][0])
            S0  = self.bounds[3][0] + torch.sigmoid(params[3])* (self.bounds[3][1]-self.bounds[3][0])

        X = (f*torch.exp(-self.b_values*Dp) + (1-f)*torch.exp(-self.b_values*D))

        return X, Dp, D, f, S0
    
class Net_triexp(nn.Module):
    def __init__(self, b_values):
        super(Net_triexp, self).__init__()
        self.bounds = [[1e-4,3e-3], [4e-3, 3e-2], [3e-2, 0.5], [0,0.4], [0,0.4], [0.6, 1.4]]
        #extend ranges to allow sigmoid to stretch over
        for idx in range(len(self.bounds)):
            self.bounds[idx][0] *= 0.7
            self.bounds[idx][1] *= 1.3
        self.b_values = b_values
        self.fc_layers = nn.ModuleList()
        for i in range(2): # 3 fully connected hidden layers
            self.fc_layers.extend([nn.Linear(len(b_values), len(b_values)),nn.BatchNorm1d(len(b_values)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values), 6))

    def forward(self, X):
        params = self.encoder(X) # Dp, Dt, Fp
        if torch.numel(torch.tensor(params.shape)) > 1:
            D = self.bounds[0][0] + torch.sigmoid(params[:, 0].unsqueeze(1)) * (self.bounds[0][1]-self.bounds[0][0])
            Dp1 = self.bounds[1][0] + torch.sigmoid(params[:, 1].unsqueeze(1)) * (self.bounds[1][1]-self.bounds[1][0])        
            Dp2 = self.bounds[2][0] + torch.sigmoid(params[:, 2].unsqueeze(1))* (self.bounds[2][1]-self.bounds[2][0])
            f1 = self.bounds[3][0] + torch.sigmoid(params[:, 3].unsqueeze(1))* (self.bounds[3][1]-self.bounds[3][0])
            f2 = self.bounds[4][0] + torch.sigmoid(params[:, 4].unsqueeze(1))* (self.bounds[4][1]-self.bounds[4][0])
            S0 = self.bounds[5][0] + torch.sigmoid(params[:, 5].unsqueeze(1))* (self.bounds[5][1]-self.bounds[5][0])
        else: 
            D = self.bounds[0][0] + torch.sigmoid(params[0]) * (self.bounds[0][1]-self.bounds[0][0])
            Dp1 = self.bounds[1][0] + torch.sigmoid(params[1]) * (self.bounds[1][1]-self.bounds[1][0])        
            Dp2 = self.bounds[2][0] + torch.sigmoid(params[2])* (self.bounds[2][1]-self.bounds[2][0])
            f1 = self.bounds[3][0] + torch.sigmoid(params[3])* (self.bounds[3][1]-self.bounds[3][0])
            f2 = self.bounds[4][0] + torch.sigmoid(params[4])* (self.bounds[4][1]-self.bounds[4][0])
            S0 = self.bounds[5][0] + torch.sigmoid(params[5])* (self.bounds[5][1]-self.bounds[5][0])

        X = (f1*torch.exp(-self.b_values*Dp1) + f2*torch.exp(-self.b_values*Dp2) + (1-f1-f2)*torch.exp(-self.b_values*D))

        return X, D, Dp1, f1, Dp2, f2, S0

def get_design_matrix(log=False):
    data_folder = os.path.join(os.getcwd(), "data_mri")
    patient_nums = os.listdir(data_folder)
    patient_nums.sort()
    X = None
    for deblurred in [False, True]:
        for patient_num in ["10", "11", "12", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']:
            for scan_type in ["pre", "post"]:

                img_series_path = os.path.join(os.getcwd(), "data_mri")
                img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + scan_type + "diffusion_imgs"))
                try:
                    with open(img_series_path, "rb") as fp:
                        img_dict_full = pickle.load(fp)
                    img_dict = img_dict_full["diffusion"]

                except:
                    continue

                b_values = list(img_dict.image_arrays.keys())

                #first for left side
                roi="l_par"
                
                try:
                    if deblurred == True:
                        img = getattr(img_dict, f"deconv_array_{roi}")
                    else:
                        img = getattr(img_dict, f"img_{roi}")
                    voxels = np.zeros((img.shape[1]*img.shape[2]*img.shape[3], 16))

                    for i in range(16):
                        voxels[:,i] = img[i,:,:,:].flatten() #/ img[0,:,:,:].flatten()
                        if log==True:
                            voxels[:,i] = np.log(voxels[:,i])
                    #remove nan rows
                    voxels_nan = np.any(np.isnan(voxels), axis=1)
                    voxels = voxels[~voxels_nan]
                    if X is None:
                        X = voxels
                    else:
                        X = np.concatenate((X, voxels), axis=0)
                    print(f"Loaded l par {deblurred} data for for {patient_num}...")
                except:
                    print(f"Could not load the deblurred image data for the left side for patient {patient_num}")
                    pass
                
                


                #for right side 
                roi="r_par"  
                try:
                    if deblurred == True:
                            img = getattr(img_dict, f"deconv_array_{roi}")
                    else:
                        img = getattr(img_dict, f"img_{roi}")
                    voxels = np.zeros((img.shape[1]*img.shape[2]*img.shape[3], 16))

                    for i in range(16):
                        voxels[:,i] = img[i,:,:,:].flatten() #/ img[0,:,:,:].flatten()
                        if log==True:
                            voxels[:,i] = np.log(voxels[:,i])
                    #remove nan rows
                    voxels_nan = np.any(np.isnan(voxels), axis=1)
                    voxels = voxels[~voxels_nan]
                    if X is None:
                        X = voxels
                    else:
                        X = np.concatenate((X, voxels), axis=0)
                    print(f"Loaded r par {deblurred} data for for {patient_num}...")
                except:
                    print(f"Could not load the deblurred image data for the right side for patient {patient_num}")
                    pass
                
                
                

                # #and for cord
                # roi="cord"
                # print(f"Loading cord data for for {patient_num}...")
                # try:
                #     if deblurred == True:
                #         img = getattr(img_dict, f"deconv_array_{roi}")
                #     else:
                #         img = getattr(img_dict, f"img_{roi}")
                #     voxels = np.zeros((img.shape[1]*img.shape[2]*img.shape[3], 16))
                #     for i in range(16):
                #         voxels[:,i] = img[i,:,:,:].flatten() / img[0,:,:,:].flatten()
                #         if log==True:
                #             voxels[:,i] = np.log(voxels[:,i])
                #     #remove nan rows
                #     voxels_nan = np.any(np.isnan(voxels), axis=1)
                #     voxels = voxels[~voxels_nan]
                #     if X is None:
                #         X = voxels
                #     else:
                #         X = np.concatenate((X, voxels), axis=0)
                # except:
                #     print(f"Could not load the deblurred image data for the cord for patient {patient_num}")
                #     pass
    #need to filter out rows that are just noise 
    noise_rows = X[:,0] < 40
    X = X[~noise_rows]

    X0 = X[:,0]
    X0 = X0[:, np.newaxis]
    X = X / X0
    

    return X, b_values

class triexp_Loss(nn.Module):
    def __init__(self, predict_s0=True):
        super(triexp_Loss, self).__init__()
        self.predict_s0 = predict_s0
    def forward(self, x_pred, x_batch, S0):
        #l1_loss = nn.L1Loss()(k,y)
        if self.predict_s0 == True:
            mse_loss = nn.MSELoss()(x_pred[:,1:],(x_batch[:,1:]/S0))+ torch.abs(torch.mean(S0-1)) #S0 is a parameter as well, normalization signal is blurry and can greatly effect all other values
        else:
            mse_loss = nn.MSELoss()(x_pred[:,1:],(x_batch[:,1:]))
        return mse_loss
    
class biexp_Loss(nn.Module):
    def __init__(self, predict_s0=True):
        super(biexp_Loss, self).__init__()
        self.predict_s0 = predict_s0
    def forward(self, x_pred, x_batch, S0):
        #l1_loss = nn.L1Loss()(k,y)
        #n = x_batch[:,1:]/S0
        if self.predict_s0 == True:
            mse_loss = nn.MSELoss()(x_pred[:,1:],(x_batch[:,1:]/S0))+ torch.abs(torch.mean(S0-1)) #S0 is a parameter as well, normalization signal is blurry and can greatly effect all other values
        else:
            mse_loss = nn.MSELoss()(x_pred[:,1:],(x_batch[:,1:]))
        return mse_loss

class adc_Loss(nn.Module):
    def __init__(self, predict_s0=True):
        super(adc_Loss, self).__init__()
        self.predict_s0 = predict_s0
    def forward(self, x_pred, x_batch, S0):
        #l1_loss = nn.L1Loss()(k,y)
        if self.predict_s0 == True:
            mse_loss = nn.MSELoss()(x_pred[:,1:],(x_batch[:,1:]/S0))+ torch.abs(torch.mean(S0-1)) #S0 is a parameter as well, normalization signal is blurry and can greatly effect all other values
        else:
            mse_loss = nn.MSELoss()(x_pred[:,1:],(x_batch[:,1:]))

        return mse_loss


def train_triexp_model(predict_s0=True):
    device = torch.device('cuda')
    X, b_values = get_design_matrix()
    
    b_values = torch.FloatTensor(b_values).to(device)
    net = Net_triexp(b_values).to(device)

    # Loss function and optimizer
    criterion = triexp_Loss(predict_s0=predict_s0)
    optimizer = optim.Adam(net.parameters(), lr = 0.00003)

    batch_size = 256
    num_batches = len(X) // batch_size
    X_train = X
    trainloader = utils.DataLoader(torch.from_numpy(X_train.astype(np.float32)),
                                    batch_size = batch_size, 
                                    shuffle = True,
                                    num_workers = 2,
                                    drop_last = True)

    # Best loss
    best = 1e16
    num_bad_epochs = 0
    patience = 5
    loss_history= []
    # Train
    for epoch in range(1000): 
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        net.train()
        running_loss = 0.

        for i, X_batch in enumerate(tqdm(trainloader), 0):
            # zero the parameter gradients
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            # forward + backward + optimize
            X_pred, _,_,_,_,_, S0 = net(X_batch)
            loss = criterion(X_pred, X_batch, S0)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss_history.append(running_loss)
        print("Loss: {}".format(running_loss))
        # early stopping
        if running_loss < best:
            print("############### Saving good model ###############################")
            final_model = net.state_dict()
            best = running_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == patience:
                print("Done, best loss: {}".format(best))
                break
    print("Done")
    with open(os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", f"triexp_loss_history_{predict_s0}"), "wb") as fp:
        pickle.dump(loss_history, fp)
    # Restore best model
    net.load_state_dict(final_model)
    torch.save({'model_state_dict': net.state_dict(), "b_values": b_values}, os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", f"triexp_with_optimizer_{predict_s0}.pt"))  

def train_biexp_model(predict_s0=True):
    #s0 == True if predicting s0 with the model.
    device = torch.device('cuda')
    X, b_values = get_design_matrix()
    
    b_values = torch.FloatTensor(b_values).to(device)
    net = Net_biexp(b_values).to(device)

    # Loss function and optimizer
    criterion = biexp_Loss(predict_s0=predict_s0)
    optimizer = optim.Adam(net.parameters(), lr = 0.00003)

    batch_size = 256
    num_batches = len(X) // batch_size
    X_train = X 
    trainloader = utils.DataLoader(torch.from_numpy(X_train.astype(np.float32)),
                                    batch_size = batch_size, 
                                    shuffle = True,
                                    num_workers = 2,
                                    drop_last = True)

    # Best loss
    best = 1e16
    num_bad_epochs = 0
    patience = 5
    loss_history = []
    # Train
    for epoch in range(1000): 
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        net.train()
        running_loss = 0.

        for i, X_batch in enumerate(tqdm(trainloader), 0):
            # zero the parameter gradients
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            # forward + backward + optimize
            X_pred, _,_,_, S0 = net(X_batch)
            loss = criterion(X_pred, X_batch, S0)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss_history.append(running_loss)
        print("Loss: {}".format(running_loss))
        # early stopping
        if running_loss < best:
            print("############### Saving good model ###############################")
            final_model = net.state_dict()
            best = running_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == patience:
                print("Done, best loss: {}".format(best))
                break
    print("Done")
    # Restore best model
    with open(os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", f"biexp_loss_history_{predict_s0}"), "wb") as fp:
        pickle.dump(loss_history, fp)

    net.load_state_dict(final_model)
    torch.save({'model_state_dict': net.state_dict(), "b_values": b_values}, os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", f"biexp_with_optimizer_{predict_s0}.pt"))  


def train_adc_model(predict_s0=True):
    device = torch.device('cuda')
    X, b_values = get_design_matrix()
    
    b_values = torch.FloatTensor(b_values).to(device)
    net = Net_ADC(b_values).to(device)

    # Loss function and optimizer
    criterion = adc_Loss(predict_s0=predict_s0)
    optimizer = optim.Adam(net.parameters(), lr = 0.00003)

    batch_size = 256
    num_batches = len(X) // batch_size
    X_train = X # exlude the b=0 value as signals are normalized
    trainloader = utils.DataLoader(torch.from_numpy(X_train.astype(np.float32)),
                                    batch_size = batch_size, 
                                    shuffle = True,
                                    num_workers = 2,
                                    drop_last = True)

    # Best loss
    best = 1e16
    num_bad_epochs = 0
    patience = 5
    loss_history = []
    # Train
    for epoch in range(1000): 
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        net.train()
        running_loss = 0.

        for i, X_batch in enumerate(tqdm(trainloader), 0):
            # zero the parameter gradients
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            # forward + backward + optimize
            X_pred, _, S0 = net(X_batch)
            loss = criterion(X_pred, X_batch, S0)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss_history.append(running_loss)
        print("Loss: {}".format(running_loss))
        # early stopping
        if running_loss < best:
            print("############### Saving good model ###############################")
            final_model = net.state_dict()
            best = running_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == patience:
                print("Done, best loss: {}".format(best))
                break
    print("Done")
    # Restore best model
    with open(os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", f"adc_loss_history_{predict_s0}"), "wb") as fp:
        pickle.dump(loss_history, fp)

    net.load_state_dict(final_model)
    torch.save({'model_state_dict': net.state_dict(), "b_values": b_values}, os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", f"adc_with_optimizer_{predict_s0}.pt"))  


def fit_triexp_voxel(voxel, deblurred=False, smoothed=False, predict_s0=True):
    #img is a 1d image (b values for voxel)
    model_params = torch.load(os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", f"triexp_with_optimizer_{predict_s0}.pt"))
    b_values = model_params["b_values"].to("cuda")
    #b_values = torch.FloatTensor(b_values)
    net = Net_triexp(b_values).to("cuda")
    net.load_state_dict(model_params["model_state_dict"])
    voxel = torch.tensor(voxel).to("cuda")

    net.eval()
    with torch.no_grad():
        _, D, Dp1, f1, Dp2, f2, S0 = net(voxel)



    return np.array([p.cpu().numpy().flatten() for p in [D, Dp1, f1, Dp2, f2, S0]])

def fit_adc_voxel(voxel, deblurred=False, smoothed=False, predict_s0=True):
    #img is a 1d image (b values for voxel)

    model_params = torch.load(os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", f"adc_with_optimizer_{predict_s0}.pt"))
    b_values = model_params["b_values"].to("cuda")
   # b_values = torch.FloatTensor(b_values)
    net = Net_ADC(b_values).to("cuda")
    net.load_state_dict(model_params["model_state_dict"])
    voxel = torch.tensor(voxel).to("cuda")
    net.eval()
    with torch.no_grad():
        _, adc, S0 = net(voxel)

        
    return np.array([p.cpu().numpy().flatten() for p in [adc, S0]])

def fit_biexp_voxel(voxel, deblurred=False, smoothed=False, predict_s0=True):
    #img is a 1d image (b values for voxel)

    model_params = torch.load(os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", f"biexp_with_optimizer_{predict_s0}.pt"))
    b_values = model_params["b_values"].to("cuda")
    #b_values = torch.FloatTensor(b_values)
    net = Net_biexp(b_values).to("cuda")
    net.load_state_dict(model_params["model_state_dict"])
    net.eval()
    
    voxel = torch.tensor(voxel).to("cuda")

    with torch.no_grad():
        _, Dp, D, f, S0 = net(voxel)
    params = np.array([p.cpu().numpy().flatten() for p in [Dp, D, f, S0]])
    # biexp_curve = biexp(b_values.cpu().numpy(), *params[:3,40])
    # fig, ax = plt.subplots()
    # ax.scatter(b_values.cpu().numpy(), voxel[40,:].cpu().numpy())#/params[3,40])
    # ax.plot(b_values.cpu().numpy(), biexp_curve)
    # plt.show()
    return params

def fit_curvature_voxel(voxel_orig, deblurred=False, smoothed=False):
    #now need to calculate curvature of the voxel (curvature = magnitude of unit tangent derivative / magnitude of tangent)
    b_values = torch.load(os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", f"biexp_with_optimizer_{deblurred}_{smoothed}.pt"))["b_values"]  #just using biexp to get b 
    voxel = np.array((voxel_orig / voxel_orig[0]))

    if np.isnan(voxel).any():
        return np.nan
    
    # spline = CubicSpline(b_values, voxel)
    b_interp = np.linspace(b_values[0], b_values[-1], 1001)
    # smooth_sig = spline(b_interp)
    # sigs_smooth = gaussian_filter1d(smooth_sig, sigma=100, mode="reflect")
    
    #kernel_sig = np.exp(kernel_sig)
    # kernel_sig[0] = 1
    # kernel_sig[-1] = voxel_orig[-1] / voxel_orig[0]

    
    spline= UnivariateSpline(b_values, voxel, k=2, s=0.01)
    sigs_smooth = spline(b_interp)

    # lowess = sm.nonparametric.lowess(voxel, b_values, frac=0.8, it=3)
    # lowess_x = list(zip(*lowess))[0]
    # lowess_y = list(zip(*lowess))[1]

    # lowess_func = interp1d(lowess_x, lowess_y, bounds_error=False)

    # sigs_smooth = lowess_func(b_interp)

    sigs_smooth = (sigs_smooth)*1000
    
    deriv = diff(sigs_smooth) 
    deriv2 = diff(deriv)
    deriv = deriv[1:] - deriv[:-1]
    curvatures = deriv2 / (1+(deriv)**2)**1.5
    max_b = b_interp[np.argmax(curvatures)+1]

    c20 = curvatures[20]   #can simply index because 1000 indices correspond to b = 1 --> 1000
    c50 = curvatures[50]
    c100 = curvatures[100]
    c150 = curvatures[150]
    c200 = curvatures[200]
    # print(max_b)
    # fig, ax = plt.subplots()
    # ax.plot(b_interp, sigs_smooth)
    # ax.scatter(b_values, voxel_orig*1000)

    # plt.show()
    return max_b, c20, c50, c100, c150, c200

def fit_auc_voxel(voxel_orig, deblurred=False, region="all"):

    #img is a 1d image (b values for voxel)
    
    b_values = torch.load(os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", f"biexp_with_optimizer.pt"))["b_values"]  #just using biexp to get b 
    b_values = b_values.cpu()
    voxel = torch.tensor((voxel_orig)).cpu()

    if np.isnan(voxel).any():
        return np.nan, np.nan, np.nan, np.nan

    #now can calculate area under curve which is log(kernel_sig) - log(segmented fit )
    integral = 0.5*(voxel[1:] + voxel[:-1]) * (b_values[1:]-b_values[:-1])   #centred discrete integral
    

    # import random
    # num = np.random.rand()* 100
    # if num > 4 and num < 5:
        # fig, ax = plt.subplots()
        # widths = (b_values[1:]-b_values[:-1])
        # ax.bar((b_values[1:]+b_values[:-1])/2, integral, width=widths, alpha=0.5, color="darkorange")
        # ax.scatter(b_values, voxel_orig, c="darkorchid")
        
        # ax.set_xlabel("b-value ($s / mm^2$)")
        # ax.set_ylabel("$S(b) / S(0)$")
        # plt.show()
        # b_range = (b_values[1:]+b_values[:-1])/2
        # fig, ax = plt.subplots()
        # widths = (b_values[1:]-b_values[:-1])
        # ax.bar(b_range[0:10], integral[0:10], width=widths[0:10], alpha=0.5, color="greenyellow")
        # ax.bar(b_range[10:13], integral[10:13], width=widths[10:13], alpha=0.5, color="crimson")
        # ax.bar(b_range[13:15], integral[13:15], width=widths[13:15], alpha=0.5, color="cornflowerblue")
        # ax.scatter(b_values, voxel_orig, c="darkorchid")
        
        # ax.set_xlabel("b-value ($s / mm^2$)")
        # ax.set_ylabel("$S(b) / S(0)$")
        # plt.show()
    

    return torch.sum(integral),torch.sum(integral[:10]),torch.sum(integral[10:13]),torch.sum(integral[-3:])


def fit_auc_voxel_discrete(voxel_orig, deblurred=False, smoothed=False):

    #img is a 1d image (b values for voxel)
    
    b_values = torch.load(os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", f"biexp_with_optimizer_{deblurred}_{smoothed}.pt"))["b_values"]  #just using biexp to get b 
    b_values = torch.tensor(b_values)
    voxel = torch.tensor((voxel_orig / voxel_orig[0]))

    if np.isnan(voxel).any():
        return np.nan
   

    integral = 0.5*(voxel[1:] + voxel[:-1]) * (b_values[1:]-b_values[:-1])   #centred discrete integral
    integral = torch.sum(integral)
    # fig, ax = plt.subplots()
    # ax.plot(b_interp, sigs_smooth)
    # ax.scatter(b_values, voxel_orig)
    # plt.show()
    #now can calculate area under curve which is log(kernel_sig) - log(segmented fit )

    return integral


def get_unit_tangent(b_values, voxel):
    #get derivative 
    ds_dx = []
    #first use central differences for inside points
    for i in range(1, voxel.size-1):
        ds_dx.append((voxel[i+1]- voxel[i-1]) / (b_values[i+1] - b_values[i-1]))
    #now add differences at ends:
    ds_dx.insert(0, (voxel[1]- voxel[0]) / (b_values[1] - b_values[0]))
    ds_dx.append((voxel[-1]- voxel[-2]) / (b_values[-1] - b_values[-2]))

    ds_dx = np.array(ds_dx)
    norm = np.linalg.norm(np.array(ds_dx))
    #now calculate the magnitude and divide
    #ds_dx /= norm
    return ds_dx, norm

def predict_all_biexp_imgs(patient_nums, scan_type, deblurred=False, smoothed=False, predict_s0=True):
    #in this function, diffusion images are smoothed, as a function of b value, using a gaussian kernel
    
    data_folder = os.path.join(os.getcwd(), "data_mri")
    for patient_num in patient_nums:#["10", "11", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']:
        print(f"Getting deblurred {deblurred} biexp imgs for {patient_num}")

        img_series_path = os.path.join(os.getcwd(), "data_mri")
        img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + scan_type + "diffusion_imgs"))
        with open(img_series_path, "rb") as fp:
            img_dict_full = pickle.load(fp)
        img_dict = img_dict_full["diffusion"]
        

        b_values = list(img_dict.image_arrays.keys())
        b_values = np.array(b_values)
        for roi in ["r_par", "l_par"]:
            try:
                if deblurred == True:
                    img_ = getattr(img_dict, f"deconv_array_{roi}")
                else:
                    img_ = getattr(img_dict, f"img_{roi}")
            except:
                print(f"Could not load the deblurred image data for {roi} for patient {patient_num}")
                continue
            img = img_ / img_[0,:,:,:]
            params_img = np.zeros((4,*img.shape[1:]))

            for z in range(img.shape[1]):
                print(f"On slice {z+1} / {img.shape[1]}")
                for y in range(img.shape[2]):
                    # for x in range(img.shape[3]):
                        
                    signal = np.array(img[:,z,y,:]).T


                    # corr, _ = spearmanr(b_values, signal)    #don't try to smooth/interpret if it is clearly noisy and not even a diffusion curve
                    # if corr > -0.6:
                    #     params_img[:,z,y,x] = np.nan
                    #     continue
                    params = fit_biexp_voxel(signal, deblurred=deblurred, smoothed=smoothed, predict_s0=predict_s0)
                    params_img[:, z,y,:] = params
            #plot_3d_image(img[0,:,:,:])
            setattr(img_dict, f"biexp_img_{roi}_{deblurred}_{smoothed}_{predict_s0}_{scan_type}", params_img)
            # plot_3d_image(params_img[0,:,:,:])
            # plot_3d_image(params_img[1,:,:,:])
            # plot_3d_image(params_img[2,:,:,:])
            # plot_3d_image(params_img[3,:,:,:])

        with open(img_series_path, "wb") as fp:
            pickle.dump(img_dict_full, fp)
    
    return

def predict_all_triexp_imgs(patient_nums, scan_type, deblurred=False, smoothed=False, predict_s0=True):
    #in this function, diffusion images are smoothed, as a function of b value, using a gaussian kernel
    data_folder = os.path.join(os.getcwd(), "data_mri")
    for patient_num in patient_nums:#["10", "11", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']:
        print(f"Getting deblurred {deblurred} triexp imgs for {patient_num}")

        img_series_path = os.path.join(os.getcwd(), "data_mri")
        img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + scan_type + "diffusion_imgs"))
        with open(img_series_path, "rb") as fp:
            img_dict_full = pickle.load(fp)
        img_dict = img_dict_full["diffusion"]

        b_values = list(img_dict.image_arrays.keys())
        b_values = np.array(b_values)
        for roi in ["l_par", "r_par"]:
            try:
                if deblurred == True:
                    img_ = getattr(img_dict, f"deconv_array_{roi}")
                else:
                    img_ = getattr(img_dict, f"img_{roi}")
            except:
                print(f"Could not load the deblurred image data for the right side for patient {patient_num}")
                continue
            img = img_ / img_[0,:,:,:]
            params_img = np.zeros((6,*img.shape[1:]))

            for z in range(img.shape[1]):
                #print(f"On slice {z+1} / {img.shape[1]}")
                for y in range(img.shape[2]):
                    #for x in range(img.shape[3]):
                    
                    signal = np.array(img[:,z,y,:]).T

                    # corr, _ = spearmanr(b_values, signal)    #don't try to smooth/interpret if it is clearly noisy and not even a diffusion curve
                    # if corr > -0.6:
                    #     params_img[:,z,y,x] = np.nan
                    #     continue
                    params = fit_triexp_voxel(signal, deblurred=deblurred, smoothed=smoothed, predict_s0=predict_s0)
                    params_img[:,z,y,:] = params

            setattr(img_dict, f"triexp_img_{roi}_{deblurred}_{smoothed}_{predict_s0}_{scan_type}", params_img)

            # plot_3d_image(img[0,:,:,:])
            # plot_3d_image(params_img[0,:,:,:])
            # plot_3d_image(params_img[1,:,:,:])
            # plot_3d_image(params_img[2,:,:,:])
            # plot_3d_image(params_img[3,:,:,:])
            # plot_3d_image(params_img[4,:,:,:])

        with open(img_series_path, "wb") as fp:
            pickle.dump(img_dict_full, fp)
    
    return

def predict_all_adc_imgs(patient_nums, scan_type, deblurred=False, smoothed=False, predict_s0=True):

    for patient_num in patient_nums:#["10", "11", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']:

        print(f"Getting deblurred {deblurred} adc imgs for {patient_num}")
        img_series_path = os.path.join(os.getcwd(), "data_mri")
        img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + scan_type + "diffusion_imgs"))
        with open(img_series_path, "rb") as fp:
            img_dict_full = pickle.load(fp)
        img_dict = img_dict_full["diffusion"]

        b_values = list(img_dict.image_arrays.keys())
        b_values = np.array(b_values)
        for roi in ["l_par", "r_par"]:
            try:
                if deblurred == True:
                    img_ = getattr(img_dict, f"deconv_array_{roi}")
                else:
                    img_ = getattr(img_dict, f"img_{roi}")
            except:
                print(f"Could not load the deblurred image data for the right side for patient {patient_num}")
                continue
            img = img_ / img_[0,:,:,:]
            adc_img = np.zeros((2,*img.shape[1:]))
            
            for z in range(img.shape[1]):
                print(f"On slice {z+1} / {img.shape[1]}")
                for y in range(img.shape[2]):
                    #for x in range(img.shape[3]):
                        
                    signal = np.array(img[:,z,y,:]).T
            

                    # corr, _ = spearmanr(b_values, signal)    #don't try to smooth/interpret if it is clearly noisy and not even a diffusion curve
                    # if corr > -0.5:
                    #     adc_img[z,y,x] = np.nan
                    #     continue
                    adc = fit_adc_voxel(signal, deblurred=deblurred, smoothed=smoothed, predict_s0=predict_s0)
                    adc_img[:,z,y,:] = adc

            # plot_3d_image(img[0,:,:,:])
            # plot_3d_image(curve_img, vmax=100)

            setattr(img_dict, f"adc_img_{roi}_{deblurred}_{smoothed}_{predict_s0}_{scan_type}", adc_img)


        with open(img_series_path, "wb") as fp:
            pickle.dump(img_dict_full, fp)
    
    return

def predict_all_curvature_imgs(patient_nums, scan_type, deblurred=False, smoothed=False, predict_s0=True):
    #in this function, diffusion images are smoothed, as a function of b value, using a gaussian kernel
    data_folder = os.path.join(os.getcwd(), "data_mri")
    for patient_num in patient_nums:#["10", "11", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']:
        print(f"Getting deblurred {deblurred} curvature imgs for {patient_num}")

        img_series_path = os.path.join(os.getcwd(), "data_mri")
        img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + scan_type + "diffusion_imgs"))
        with open(img_series_path, "rb") as fp:
            img_dict_full = pickle.load(fp)
        img_dict = img_dict_full["diffusion"]

        b_values = list(img_dict.image_arrays.keys())
        b_values = np.array(b_values)
        for roi in ["l_par", "r_par"]:
            try:
                if deblurred == True:
                    img = getattr(img_dict, f"deconv_array_{roi}")
                else:
                    img = getattr(img_dict, f"img_{roi}")
                s0_img=None
                if predict_s0 == True:
                    s0_img = getattr(img_dict, f"triexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")[5,...]
            except:
                print(f"Could not load the deblurred image data for the right side for patient {patient_num}")
                continue

            curve_img = np.zeros(img.shape[1:])
            curve20_img = np.zeros(img.shape[1:])
            curve50_img = np.zeros(img.shape[1:])
            curve100_img = np.zeros(img.shape[1:])
            curve150_img = np.zeros(img.shape[1:])
            curve200_img = np.zeros(img.shape[1:])

            for z in range(img.shape[1]):
                print(f"On slice {z+1} / {img.shape[1]}")
                for y in range(img.shape[2]):
                
                    if predict_s0 == False:
                        signal = np.array(img[:,z,y,:]) / np.array(img[0,z,y,:])
                    else:
                        signal = np.array(img[:,z,y,:]) / s0_img[z,y,:]

                    b_max, c20, c50, c100, c150, c200 = fit_curvature_voxel(signal, deblurred=deblurred, smoothed=smoothed)
                    curve_img[z,y,:] = b_max
                    curve20_img[z,y,:] = c20
                    curve50_img[z,y,:] = c50
                    curve100_img[z,y,:] = c100
                    curve150_img = c150
                    curve200_img = c200

            # plot_3d_image(img[0,:,:,:])
            # plot_3d_image(curve_img, vmax=100)

            setattr(img_dict, f"curve_img_{roi}_{deblurred}_{smoothed}_{scan_type}", curve_img)
            setattr(img_dict, f"c20_img_{roi}_{deblurred}_{smoothed}_{scan_type}", curve20_img)
            setattr(img_dict, f"c50_img_{roi}_{deblurred}_{smoothed}_{scan_type}", curve50_img)
            setattr(img_dict, f"c100_img_{roi}_{deblurred}_{smoothed}_{scan_type}", curve100_img)
            setattr(img_dict, f"c100_img_{roi}_{deblurred}_{smoothed}_{scan_type}", curve100_img)
            setattr(img_dict, f"c150_img_{roi}_{deblurred}_{smoothed}_{scan_type}", curve150_img)
            setattr(img_dict, f"c200_img_{roi}_{deblurred}_{smoothed}_{scan_type}", curve200_img)

        with open(img_series_path, "wb") as fp:
            pickle.dump(img_dict_full, fp)
    
    return

def predict_all_auc_imgs(patient_nums, scan_type, deblurred=False, smoothed=False, predict_s0=False):
    #in this function, diffusion images are smoothed, as a function of b value, using a gaussian kernel
    data_folder = os.path.join(os.getcwd(), "data_mri")
    for patient_num in patient_nums:#["19", "20", "21", "22", "23" ]: #["10", "11", "12", "13", "15", "16", "18"]:

        img_series_path = os.path.join(os.getcwd(), "data_mri")
        img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + scan_type + "diffusion_imgs"))
        with open(img_series_path, "rb") as fp:
            img_dict_full = pickle.load(fp)
        img_dict = img_dict_full["diffusion"]

        b_values = list(img_dict.image_arrays.keys())
        b_values = np.array(b_values)
        for roi in ["l_par", "r_par"]:
            try:
                if deblurred == True:
                    img_ = getattr(img_dict, f"deconv_array_{roi}")
                else:
                    img_ = getattr(img_dict, f"img_{roi}")
                s0_img=None
                if predict_s0 == True:
                    s0_img = getattr(img_dict, f"triexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")[5,...]
            except:
                print(f"Could not load the deblurred image data of {roi} for patient {patient_num}")
                continue

            img = img_ / img_[0,:,:,:]

            auc_img = np.zeros(img.shape[1:])
            #auc_img_discrete = np.zeros(img.shape[1:])
            auc_l_img = np.zeros(img.shape[1:])
            auc_m_img = np.zeros(img.shape[1:])
            auc_h_img = np.zeros(img.shape[1:])
            

            for z in range(img.shape[1]):
                print(f"On slice {z+1} / {img.shape[1]}")
                for y in range(img.shape[2]):
                    for x in range(img.shape[3]):
                        signal = img[:,z,y,x]
                        if predict_s0 == True:
                            signal = np.array(img[:,z,y,x]) / s0_img[z,y,x]

                        params = fit_auc_voxel(signal, deblurred=deblurred, region="all")
                        auc_img[z,y,x] = params[0]
                        auc_l_img[z,y,x] = params[1]
                        auc_m_img[z,y,x] = params[2]
                        auc_h_img[z,y,x] = params[3]


            # plot_3d_image(img[0,:,:,:])
            # plot_3d_image(auc_img)

            setattr(img_dict, f"auc_img_{roi}_{deblurred}_{smoothed}_{predict_s0}_{scan_type}", auc_img)
            setattr(img_dict, f"auc_l_img_{roi}_{deblurred}_{smoothed}_{predict_s0}_{scan_type}", auc_l_img)
            setattr(img_dict, f"auc_m_img_{roi}_{deblurred}_{smoothed}_{predict_s0}_{scan_type}", auc_m_img)
            setattr(img_dict, f"auc_h_img_{roi}_{deblurred}_{smoothed}_{predict_s0}_{scan_type}", auc_h_img)

        

 

        with open(img_series_path, "wb") as fp:
            pickle.dump(img_dict_full, fp)
    
    return

def view_blur_deblur_and_conv():
    for patient_num in ["10", "11", "12", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']:
        img_series_path = os.path.join(os.getcwd(), "data_mri")
        img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + "pre" + "diffusion_imgs"))
        with open(img_series_path, "rb") as fp:
            img_dict_full = pickle.load(fp)
        img_dict = img_dict_full["diffusion"]

        b_values = list(img_dict.image_arrays.keys())
        b_values = np.array(b_values)
        for roi in ["l_par", "r_par"]:
            try:

                img_deblur = getattr(img_dict, f"deconv_array_{roi}")

                kernel = getattr(img_dict, f"kernel_{roi}")
                
                img_conv = getattr(img_dict, f"deconv_array_blurred_{roi}")

                
                img_orig = getattr(img_dict, f"img_{roi}")
            except:
                continue
            img_conv /= np.amax(img_conv) 
            img_deblur /= np.amax(img_deblur)
            kernel /= np.amax(kernel)
            img_orig /= np.amax(img_orig)
            fig, ax = plt.subplots(ncols=3)

            ax[0].imshow(img_orig[0,15,...])
            ax[1].imshow(img_deblur[0,15,...])
            ax[2].imshow(np.sum(kernel, axis=0))
           # ax[3].imshow(img_conv[0,15,...])
            ax[0].set_title(f"Original", fontsize=20)
            ax[1].set_title(f"Deblurred", fontsize=20)
            ax[2].set_title(f"Kernel", fontsize=20)
            #ax[3].set_title(f"Deblurred * Kernel", fontsize=20)
            for i in range(3):
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                ax[i].set_xlabel('')
                ax[i].set_ylabel('')
            plt.show()

def view_all_predicted_imgs(patient_nums, scan_type, deblurred=True):
    for patient_num in patient_nums:


        img_series_path = os.path.join(os.getcwd(), "data_mri")
        img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + scan_type + "diffusion_imgs"))
        with open(img_series_path, "rb") as fp:
            img_dict_full = pickle.load(fp)
        img_dict = img_dict_full["diffusion"]

        b_values = list(img_dict.image_arrays.keys())
        b_values = np.array(b_values)
        for roi in ["r_par", "l_par"]:
            try:
                if deblurred == True:

                    img = getattr(img_dict, f"deconv_array_{roi}")
                    conv_img = getattr(img_dict, f"deconv_array_blurred_{roi}")
                    img_orig = getattr(img_dict, f"img_{roi}")
                    noise = getattr(img_dict, f"noise_{roi}")
                    kernel = getattr(img_dict, f"kernel_{roi}")
                    plot_3d_image(img_orig[0,:,:,:])
                    plot_3d_image(conv_img[0,:,:,:])
                    plot_3d_image(img[0,:,:,:])
                    plot_3d_image(noise[0,:,:,:])
                    plot_3d_image(kernel[:,:,:])
                else:
                   
                    img = getattr(img_dict, f"img_{roi}")
                    plot_3d_image(img[0,:,:,:])

                # auc_img = getattr(img_dict, f"auc_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                # adc_img = getattr(img_dict, f"adc_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                # curve_img = getattr(img_dict, f"curve_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                # c20_img = getattr(img_dict, f"c20_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                # c50_img = getattr(img_dict, f"c50_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                # c100_img = getattr(img_dict, f"c100_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                # biexp_img = getattr(img_dict, f"biexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                # triexp_img = getattr(img_dict, f"triexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                
                
                # plot_3d_image(biexp_img[0,:,:,:])
                # plot_3d_image(biexp_img[1,:,:,:])
                # plot_3d_image(biexp_img[2,:,:,:])
                # plot_3d_image(triexp_img[0,:,:,:])
                # plot_3d_image(triexp_img[1,:,:,:])
                # plot_3d_image(triexp_img[2,:,:,:])
                # plot_3d_image(triexp_img[3,:,:,:])
                # plot_3d_image(triexp_img[4,:,:,:])
                # adc_img = np.clip(adc_img, a_min=0, a_max=0.004)
                # auc_img = np.clip(auc_img, a_min=300, a_max=900)
                # curve_img = np.clip(curve_img, a_min=0,a_max=200)
                # plot_3d_image(adc_img)
                # plot_3d_image(auc_img)
                # plot_3d_image(curve_img)
                #plot_3d_image(c100_img)
                
            except Exception:

                print(f"Could not load image data of {roi} for patient {patient_num}")
                continue

def clean_parameter_predictions():
    for scan_type in ["pre", "post"]:
        if scan_type == "pre":
            patient_nums = ["11", "10", "12", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']
        else:
            patient_nums = ["10","11","12","13", "16"]
        data_folder = os.path.join(os.getcwd(), "data_mri")

        delete_imgs = ["curv", "adc", "auc", "biexp", "triexp", "c100", "c200", "c50", "c150", "c20", "cord", "smooth"]
        for patient_num in patient_nums:#["10", "11", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']:
            print(f"Getting imgs for {patient_num}")

            img_series_path = os.path.join(os.getcwd(), "data_mri")
            img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + scan_type + "diffusion_imgs"))
            with open(img_series_path, "rb") as fp:
                img_dict_full = pickle.load(fp)
            img_dict = img_dict_full["diffusion"]

            for attr_name in list(vars(img_dict)):
                for delete_str in delete_imgs:
                    if delete_str in attr_name:
                        delattr(img_dict, attr_name)
                        break

            with open(img_series_path, "wb") as fp:
                pickle.dump(img_dict_full, fp)


def get_triexp_param_heatmap(deblurred=False):
    data_folder = os.path.join(os.getcwd(), "data_mri")
    for patient_num in ["10"]:

        img_series_path = os.path.join(os.getcwd(), "data_mri")
        img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + "pre" + "diffusion_imgs"))
        with open(img_series_path, "rb") as fp:
            img_dict_full = pickle.load(fp)
        img_dict = img_dict_full["diffusion"]
        with open(os.path.join(data_folder, str("sivim" + patient_num + "_pre_MR_mask_dict")), "rb") as fp:
            mask_dict = pickle.load(fp)

        b_values = list(img_dict.image_arrays.keys())
        b_values = np.array(b_values)

        try:
            
            with open(os.path.join(os.getcwd(),"models_ivim",  f"{patient_num}_model_stuff_right.txt"), "rb") as fp:    #right
                [out_x, out_k, img_orig, _, _, _, _] = pickle.load(fp)  
            if deblurred == True:
                img = out_x
            else:
                img = img_orig
        except:
            print(f"Could not load the deblurred image data for the right side for patient {patient_num}")
            continue
        #now can predict img:
        from blind_deconv_analysis_ivim import plot_signal_curve
        voxel = img[:,15,30,30]
        voxel = np.array(voxel) / voxel[0]
        voxel = torch.tensor(voxel)
        #plot_signal_curve(b_values, img[:,15,30,30])
        triexp_params = fit_triexp_voxel(img[:,15,30,30])

        criterion = nn.MSELoss()
        loss_vals = np.zeros((100,100))
        f_grid = np.linspace(0.1,0.3,100)
        Dp_grid = np.linspace(0.002,0.01, 100)
        for i, f in enumerate(f_grid):
            for j, Dp in enumerate(Dp_grid):
                triexp_sigs = triexp(b_values, Dp, triexp_params[1], f)
                loss = criterion(voxel,torch.tensor(triexp_sigs))
                loss_vals[i,j] = -np.log(loss.item())

        fig, ax = plt.subplots()
        im = ax.imshow(loss_vals, cmap="plasma")

        x_ticks = np.arange(0, len(Dp_grid), 10)
        x_tick_labels = [f'{val/100*(0.01-0.002) + 0.002:.3f}' for val in x_ticks]
        y_ticks = np.arange(0, len(f_grid), 10)
        y_tick_labels = [f'{val/100*(0.3-0.1)+0.1:.2f}' for val in y_ticks]

        ax.set_xticks(x_ticks, x_tick_labels)
        ax.set_yticks(y_ticks, y_tick_labels)
        plt.colorbar(im, label='negative log loss')

        plt.show()
        #biexp_sigs = biexp(b_values, *biexp_params)

    
    return

def get_biexp_param_heatmap(deblurred=False):
    data_folder = os.path.join(os.getcwd(), "data_mri")
    for patient_num in ["10"]:

        img_series_path = os.path.join(os.getcwd(), "data_mri")
        img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + "pre" + "diffusion_imgs"))
        with open(img_series_path, "rb") as fp:
            img_dict_full = pickle.load(fp)
        img_dict = img_dict_full["diffusion"]
        with open(os.path.join(data_folder, str("sivim" + patient_num + "_pre_MR_mask_dict")), "rb") as fp:
            mask_dict = pickle.load(fp)

        b_values = list(img_dict.image_arrays.keys())
        b_values = np.array(b_values)

        try:
            
            with open(os.path.join(os.getcwd(),"models_ivim",  f"{patient_num}_model_stuff_left.txt"), "rb") as fp:    #right
                [out_x, out_k, img_orig, _, _, _, _] = pickle.load(fp)  
            if deblurred == True:
                img = out_x
            else:
                img = img_orig
        except:
            print(f"Could not load the deblurred image data for the right side for patient {patient_num}")
            continue
        #now can predict img:
        from blind_deconv_analysis_ivim import plot_signal_curve
        #[:,15,20,20]
        #[:,10,20,20]
        #[:,10,35,30]
        voxel = deepcopy(img[:,10,20,20])
        voxel = np.array(voxel) / voxel[0]
        voxel = torch.tensor(voxel)

        biexp_params = fit_biexp_voxel(voxel)
        plot_signal_curve(b_values, voxel, fit_params=biexp_params)

        criterion = nn.MSELoss()
        loss_vals = np.zeros((150,150))
        f_grid = np.linspace(0.01,0.3,150)
        Dp_grid = np.linspace(0.001,0.5, 150)
        for i, f in enumerate(f_grid):
            for j, Dp in enumerate(Dp_grid):
                biexp_sigs = biexp(b_values, Dp, biexp_params[1], f)
                loss = criterion(voxel,torch.tensor(biexp_sigs))
                loss_vals[j,i] = -np.log(loss.item())

        fig, ax = plt.subplots(figsize=(20,20))
        im = ax.imshow(loss_vals, cmap="plasma")

        x_ticks = np.linspace(0, len(Dp_grid), 5)
        x_tick_labels = [f'{val/100*(0.5-0.001) + 0.001:.3f}' for val in x_ticks]
        y_ticks = np.linspace(0, len(f_grid), 10)
        y_tick_labels = [f'{val/100*(0.3-0.01)+0.01:.2f}' for val in y_ticks]
        ax.tick_params(axis='both', labelsize=20)
        ax.set_xticks(x_ticks, x_tick_labels)
        ax.set_yticks(y_ticks, y_tick_labels)
        ax.set_ylabel("$f$", fontsize=30)
        ax.set_xlabel("$D*$", fontsize=30)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        cbar = plt.colorbar(im, label='negative log loss')
        cbar.ax.tick_params(labelsize=15)
        plt.show()
        #biexp_sigs = biexp(b_values, *biexp_params)

    
    return

def make_curvature_plot(deblurred=False):
    data_folder = os.path.join(os.getcwd(), "data_mri")
    for patient_num in ["10"]:

        img_series_path = os.path.join(os.getcwd(), "data_mri")
        img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + "pre" + "diffusion_imgs"))
        with open(img_series_path, "rb") as fp:
            img_dict_full = pickle.load(fp)
        img_dict = img_dict_full["diffusion"]
        with open(os.path.join(data_folder, str("sivim" + patient_num + "_pre_MR_mask_dict")), "rb") as fp:
            mask_dict = pickle.load(fp)

        b_values = list(img_dict.image_arrays.keys())
        b_values = np.array(b_values)

        try:
            
            with open(os.path.join(os.getcwd(),"models_ivim",  f"{patient_num}_model_stuff_right.txt"), "rb") as fp:    #right
                [out_x, out_k, img_orig, _, _, _, _] = pickle.load(fp)  
            if deblurred == True:
                img = out_x
            else:
                img = img_orig
        except:
            print(f"Could not load the deblurred image data for the right side for patient {patient_num}")
            continue
        #now can predict img:
        from blind_deconv_analysis_ivim import plot_signal_curve
        voxel = np.array(img[:,10,30,30])
        voxel = voxel / voxel[0]
        plot_signal_curve(b_values, voxel)
        curvature = fit_curvature_voxel(b_values, voxel)
        fig, ax = plt.subplots()
        ax.scatter(b_values, voxel)
        ax.scatter(b_values, curvature)

        plt.show()
    return

def apply_kernel_smoothing(deblurred=True):
    #in this function, diffusion images are smoothed, as a function of b value, using a gaussian kernel
    data_folder = os.path.join(os.getcwd(), "data_mri")
    for patient_num in ["10", "11", "12", "13", "15", "16", "18"]:

        img_series_path = os.path.join(os.getcwd(), "data_mri")
        img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + "pre" + "diffusion_imgs"))
        with open(img_series_path, "rb") as fp:
            img_dict_full = pickle.load(fp)
        img_dict = img_dict_full["diffusion"]

        b_values = list(img_dict.image_arrays.keys())
        b_values = np.array(b_values)
        for roi in ["l_par", "r_par",  "cord"]:
            try:
                if deblurred == True:
                    img = getattr(img_dict, f"deconv_array_{roi}")
                else:
                    img = getattr(img_dict, f"img_{roi}")
            except:
                print(f"Could not load the deblurred image data for the right side for patient {patient_num}")
                continue
            #now need to apply gaussian kernel smoothing to this image. 
            #plot_3d_image(img[0,:,:,:])
            smooth_img = np.zeros_like(img)
            b_interp = np.linspace(b_values[0], b_values[-1], 1000)
            for z in range(img.shape[1]):
                print(f"On slice {z+1} / {img.shape[1]}")
                for y in range(img.shape[2]):
                    for x in range(img.shape[3]):
                        
                        signal = np.array(img[:,z,y,x]) / np.array(img[0,z,y,x])
                        if np.isnan(signal).any():
                            continue
                        corr, _ = spearmanr(b_values, signal)    #don't try to smooth/interpret if it is clearly noisy and not even a diffusion curve
                        if corr > -0.75:
                            continue
                        spline = CubicSpline(b_values, signal)

                        smooth_sig = spline(b_interp)
                        kernel_sig = gaussian_filter1d(smooth_sig, sigma=15)
                        kernel_sig[0] = 1
                        kernel_sig_orig_bs = np.interp(b_values, b_interp, kernel_sig)
                        kernel_sig_orig_bs[0] = 1
                        # plot_signal_curve(b_values, signal)
                        # plot_signal_curve(b_interp, smooth_sig)
                        # plot_signal_curve(b_interp, kernel_sig)
                        # plot_signal_curve(b_values, kernel_sig_orig_bs)
                        smooth_img[:,z,y,x] = kernel_sig_orig_bs* img[0,z,y,x]
            # plot_3d_image(img[1,:,:,:])     
            # plot_3d_image(smooth_img[1,:,:,:])    
            # plot_3d_image(smooth_img[2,:,:,:])   
            # plot_3d_image(smooth_img[4,:,:,:])      

            if deblurred == True:
                setattr(img_dict, f"deconv_array_{roi}_smooth", smooth_img)
            else:
                setattr(img_dict, f"img_{roi}_smooth", smooth_img)    

        with open(img_series_path, "wb") as fp:
            pickle.dump(img_dict_full, fp)
    
    return


def get_all_parotid_param_and_dose_stats(deblurred=True, smoothed=False, scan_type="pre", predict_s0=True):
    
    #dose stats already obtained 
    path = "/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep"
    with open(os.path.join(path, "dose_data.txt"), "rb") as fp:
        dose_stats = pickle.load(fp)
    #need to get all the different params for each different img type

    all_stats = {}
    

    #triexp order D, Dp1, Dp2, f1, f2
    #biexp order pD, D, f


    if scan_type == "pre":
        patient_nums = ["11", "10", "12", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']
    else:
        patient_nums = ["10","11","12","13", "16"]

    for patient_num in patient_nums:
        
        all_stats[patient_num] = {}



        img_series_path = os.path.join(os.getcwd(), "data_mri")
        img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + scan_type + "diffusion_imgs"))
        with open(img_series_path, "rb") as fp:
            img_dict_full = pickle.load(fp)
        img_dict = img_dict_full["diffusion"]

        

        biexp_cord_avgs = [[[],[]],[[],[]],[[],[]]] #for cord, will be whole gland values only
        triexp_cord_avgs = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]
        auc_cord_avgs = [[],[]]
        curvature_cord_avgs = [[],[]]
        c20_cord_avgs = [[],[]]
        c50_cord_avgs = [[],[]]
        c100_cord_avgs = [[],[]]
        adc_cord_avgs = [[],[]]

        with open(os.path.join(os.path.join(os.getcwd(), "data_mri"), str("sivim" + patient_num + "_" + scan_type + "_MR_mask_dict")), "rb") as fp:
            mask_dict = pickle.load(fp)
        
       

        roi_masks = {}
        for roi in ["l_par", "r_par"]:
            if roi not in mask_dict.keys():
                continue
            all_stats[patient_num][roi] = {}
            #load different parameter images
            
            
            if "cord" not in roi.lower():
                all_stats[patient_num][roi] = {}
                #subsegment as needed
                Chopper.organ_chopper(mask_dict[roi], [0,0,2], name="segmented_contours_si")
                Chopper.organ_chopper(mask_dict[roi], [0,2,0], name="segmented_contours_ap")
                Chopper.organ_chopper(mask_dict[roi], [2,0,0], name="segmented_contours_ml")
                if roi == "r_par":
                    mask_dict[roi].segmented_contours_ml.reverse()   #sort medial to lateral

                coords_array = getattr(img_dict, f"deconv_coords_{roi}")
                #plot_3d_image(mask_dict[roi].mask_deconv)
                get_contour_masks.get_deconv_structure_masks(coords_array, mask_dict[roi])
            else:

                try:
                    whole_mask = mask_dict[roi].mask_deconv.astype(bool)
                    whole_mask = mask_dict[roi].mask_deconv.astype(bool)
                    auc_img= getattr(img_dict, f"auc_img_{roi}_{deblurred}_{smoothed}_{False}_{scan_type}")
                    auc_l_img = getattr(img_dict, f"auc_l_img_{roi}_{deblurred}_{smoothed}_{False}_{scan_type}")
                    auc_m_img = getattr(img_dict, f"auc_m_img_{roi}_{deblurred}_{smoothed}_{False}_{scan_type}")
                    auc_h_img = getattr(img_dict, f"auc_h_img_{roi}_{deblurred}_{smoothed}_{False}_{scan_type}")

                    auc_s0_img = getattr(img_dict, f"auc_img_{roi}_{deblurred}_{smoothed}_{True}_{scan_type}")    #version where uses exponential s0 prediction
                    auc_s0_l_img= getattr(img_dict, f"auc_l_img_{roi}_{deblurred}_{smoothed}_{True}_{scan_type}")
                    auc_s0_m_img = getattr(img_dict, f"auc_m_img_{roi}_{deblurred}_{smoothed}_{True}_{scan_type}")
                    auc_s0_h_img = getattr(img_dict, f"auc_h_img_{roi}_{deblurred}_{smoothed}_{True}_{scan_type}")
                    adc_img = getattr(img_dict, f"adc_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    curve_img = getattr(img_dict, f"curve_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    c20_img = getattr(img_dict, f"c20_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    c50_img = getattr(img_dict, f"c50_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    c100_img = getattr(img_dict, f"c100_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    triexp_img = getattr(img_dict, f"triexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    biexp_img = getattr(img_dict, f"biexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")

                    if deblurred == True:
                        auc_img_orig = getattr(img_dict, f"auc_img_{roi}_{False}_{smoothed}_{False}_{scan_type}")
                        auc_l_img_orig = getattr(img_dict, f"auc_l_img_{roi}_{False}_{smoothed}_{False}_{scan_type}")
                        auc_m__img_orig = getattr(img_dict, f"auc_m_img_{roi}_{False}_{smoothed}_{False}_{scan_type}")
                        auc_h_img_orig = getattr(img_dict, f"auc_h_img_{roi}_{False}_{smoothed}_{False}_{scan_type}")

                        auc_s0_img_orig = getattr(img_dict, f"auc_img_{roi}_{False}_{smoothed}_{True}_{scan_type}")    #version where uses exponential s0 prediction
                        auc_s0_l_img_orig = getattr(img_dict, f"auc_l_img_{roi}_{False}_{smoothed}_{True}_{scan_type}")
                        auc_s0_m__img_orig = getattr(img_dict, f"auc_m_img_{roi}_{False}_{smoothed}_{True}_{scan_type}")
                        auc_s0_h_img_orig = getattr(img_dict, f"auc_h_img_{roi}_{False}_{smoothed}_{True}_{scan_type}")

                        adc_img_orig = getattr(img_dict, f"adc_img_{roi}_{False}_{smoothed}_{scan_type}")
                        curve_img_orig = getattr(img_dict, f"curve_img_{roi}_{False}_{smoothed}_{scan_type}")
                        c20_img_orig = getattr(img_dict, f"c20_img_{roi}_{False}_{smoothed}_{scan_type}")
                        c50_img_orig = getattr(img_dict, f"c50_img_{roi}_{False}_{smoothed}_{scan_type}")
                        c100_img_orig = getattr(img_dict, f"c100_img_{roi}_{False}_{smoothed}_{scan_type}")
                        triexp_img_orig = getattr(img_dict, f"triexp_img_{roi}_{False}_{smoothed}_{scan_type}")
                        biexp_img_orig = getattr(img_dict, f"biexp_img_{roi}_{False}_{smoothed}_{scan_type}")

                        auc_img[auc_img_orig == np.nan] = np.nan
                        auc_l_img[auc_img_orig == np.nan] = np.nan
                        auc_m_img[auc_img_orig == np.nan] = np.nan
                        auc_h_img[auc_img_orig == np.nan] = np.nan

                        auc_s0_img[auc_img_orig == np.nan] = np.nan
                        auc_s0_l_img[auc_img_orig == np.nan] = np.nan
                        auc_s0_m_img[auc_img_orig == np.nan] = np.nan
                        auc_s0_h_img[auc_img_orig == np.nan] = np.nan


                        adc_img[auc_img_orig == np.nan] = np.nan
                        curve_img[auc_img_orig == np.nan] = np.nan 
                        c20_img[auc_img_orig == np.nan] = np.nan
                        c50_img[auc_img_orig == np.nan] = np.nan
                        c100_img[auc_img_orig == np.nan] = np.nan
                        triexp_img[auc_img_orig == np.nan] = np.nan
                        biexp_img[auc_img_orig == np.nan] = np.nan 
                except:
                    continue
                #get biexp params:
                for i in range(3):
                    # plot_3d_image(biexp_img[i,...])
                    # plot_3d_image(whole_mask)
                    img = biexp_img[i,...]
                    vals = img[whole_mask]
                    mean = np.nanmean(vals)
                    std = np.nanstd(vals)
                    biexp_cord_avgs[i][0].append(mean)   #add all 3 biexp params
                    biexp_cord_avgs[i][1].append(std)


                all_stats[patient_num][roi]["biexps"] = biexp_cord_avgs

                


                #get triexp params:
                for i in range(5):

                    img = triexp_img[i,...]
                    vals = img[whole_mask]
                    mean = np.nanmean(vals)
                    std = np.nanstd(vals)
                    triexp_cord_avgs[i][0].append(mean)    #add all 5 triexp params
                    triexp_cord_avgs[i][1].append(std)

                all_stats[patient_num][roi]["triexp"] = triexp_cord_avgs


            #now auc image

                img = auc_img
                vals = img[whole_mask]
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                auc_cord_avgs[0].append(mean)  
                auc_cord_avgs[1].append(std)
                all_stats[patient_num][roi]["aucs"] = auc_cord_avgs


            #now ADC image

                img = adc_img
                vals = img[whole_mask]
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                adc_cord_avgs[0].append(mean)   
                adc_cord_avgs[1].append(std)
                all_stats[patient_num][roi]["adcs"] = adc_cord_avgs


            #now curvature image
                img = curve_img
                vals = img[whole_mask]
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                curvature_cord_avgs[0].append(mean)  
                curvature_cord_avgs[1].append(std)
                all_stats[patient_num][roi]["curvatures"] = curvature_cord_avgs


                img = c20_img
                vals = img[whole_mask]
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                c20_cord_avgs[0].append(mean)  
                c20_cord_avgs[1].append(std)
                all_stats[patient_num][roi]["c20s"] = c20_cord_avgs

                img = c50_img
                vals = img[whole_mask]
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                c50_cord_avgs[0].append(mean)  
                c50_cord_avgs[1].append(std)
                all_stats[patient_num][roi]["c50s"] = c50_cord_avgs

                img = c100_img
                vals = img[whole_mask]
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                c100_cord_avgs[0].append(mean)  
                c100_cord_avgs[1].append(std)
                all_stats[patient_num][roi]["c100s"] = c100_cord_avgs
                
                
                
                

        for roi in ["l_par", "r_par"]:


            try:
                whole_mask = mask_dict[roi].mask_deconv.astype(bool)
                auc_img = getattr(img_dict, f"auc_img_{roi}_{deblurred}_{smoothed}_{False}_{scan_type}")
                auc_l_img = getattr(img_dict, f"auc_l_img_{roi}_{deblurred}_{smoothed}_{False}_{scan_type}")
                auc_m_img = getattr(img_dict, f"auc_m_img_{roi}_{deblurred}_{smoothed}_{False}_{scan_type}")
                auc_h_img = getattr(img_dict, f"auc_h_img_{roi}_{deblurred}_{smoothed}_{False}_{scan_type}")
                auc_s0_img = getattr(img_dict, f"auc_img_{roi}_{deblurred}_{smoothed}_{True}_{scan_type}")    #version where uses exponential s0 prediction
                auc_s0_l_img = getattr(img_dict, f"auc_l_img_{roi}_{deblurred}_{smoothed}_{True}_{scan_type}")
                auc_s0_m_img = getattr(img_dict, f"auc_m_img_{roi}_{deblurred}_{smoothed}_{True}_{scan_type}")
                auc_s0_h_img = getattr(img_dict, f"auc_h_img_{roi}_{deblurred}_{smoothed}_{True}_{scan_type}")

                adc_img = getattr(img_dict, f"adc_img_{roi}_{deblurred}_{smoothed}_{predict_s0}_{scan_type}")
                triexp_img = getattr(img_dict, f"triexp_img_{roi}_{deblurred}_{smoothed}_{predict_s0}_{scan_type}")
                biexp_img = getattr(img_dict, f"biexp_img_{roi}_{deblurred}_{smoothed}_{predict_s0}_{scan_type}")

                auc_img_orig = getattr(img_dict, f"auc_img_{roi}_{False}_{smoothed}_{False}_{scan_type}")
                auc_l_img_orig = getattr(img_dict, f"auc_img_{roi}_{False}_{smoothed}_{False}_{scan_type}")
                auc_m_img_orig = getattr(img_dict, f"auc_img_{roi}_{False}_{smoothed}_{False}_{scan_type}")
                auc_h_img_orig = getattr(img_dict, f"auc_img_{roi}_{False}_{smoothed}_{False}_{scan_type}")
                auc_s0_img_orig = getattr(img_dict, f"auc_img_{roi}_{False}_{smoothed}_{True}_{scan_type}")
                auc_s0_l_img_orig = getattr(img_dict, f"auc_img_{roi}_{False}_{smoothed}_{True}_{scan_type}")
                auc_s0_m_img_orig = getattr(img_dict, f"auc_img_{roi}_{False}_{smoothed}_{True}_{scan_type}")
                auc_s0_h_img_orig = getattr(img_dict, f"auc_img_{roi}_{False}_{smoothed}_{True}_{scan_type}")
                adc_img_orig = getattr(img_dict, f"adc_img_{roi}_{False}_{smoothed}_{predict_s0}_{scan_type}")
                triexp_img_orig = getattr(img_dict, f"triexp_img_{roi}_{False}_{smoothed}_{predict_s0}_{scan_type}")
                biexp_img_orig = getattr(img_dict, f"biexp_img_{roi}_{False}_{smoothed}_{predict_s0}_{scan_type}")
                img_orig = getattr(img_dict, f"img_{roi}")

                # thres = 150
                # auc_img[img_orig[0,...] < thres] = np.nan
                # auc_l_img[img_orig[0,...] < thres] = np.nan
                # auc_m_img[img_orig[0,...] < thres] = np.nan
                # auc_h_img[img_orig[0,...] < thres] = np.nan
                # auc_s0_img[img_orig[0,...] < thres] = np.nan
                # auc_s0_l_img[img_orig[0,...] < thres] = np.nan
                # auc_s0_m_img[img_orig[0,...] < thres] = np.nan
                # auc_s0_h_img[img_orig[0,...] < thres] = np.nan
                # for i in range(2):         
                #     adc_img[i,...][[img_orig[0,...] < thres]] = np.nan
                # for i in range(5):
                #     triexp_img[i,...][img_orig[0,...] < thres] = np.nan
                # for i in range(3):
                #     biexp_img[i,...][img_orig[0,...] < thres] = np.nan
            except:
                continue

            biexp_roi_avgs = [[[],[]],[[],[]],[[],[]]] #list for avg and stds inside. will put all the seg values in here in order, and whole gland value as first entry
            triexp_roi_avgs = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]
            auc_roi_avgs = [[],[]]
            auc_l_roi_avgs = [[],[]]
            auc_m_roi_avgs = [[],[]]
            auc_h_roi_avgs = [[],[]]
            auc_s0_roi_avgs = [[],[]]
            auc_s0_l_roi_avgs = [[],[]]
            auc_s0_m_roi_avgs = [[],[]]
            auc_s0_h_roi_avgs = [[],[]]
            adc_roi_avgs = [[],[]]

            #first add whole gland stats
            whole_mask = mask_dict[roi].mask_deconv.astype(bool)
            
            for i in range(3):
                img = biexp_img[i,...]
                # img[~whole_mask] = 0
                # plot_3d_image(img)
            #get biexp params:
                vals = img[whole_mask]
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                biexp_roi_avgs[i][0].append(mean)   #add all 3 biexp params
                biexp_roi_avgs[i][1].append(std)

            #get triexp params:
            for i in range(5):
                img = triexp_img[i,...]
                vals = img[whole_mask]
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                triexp_roi_avgs[i][0].append(mean)    #add all 5 triexp params
                triexp_roi_avgs[i][1].append(std)

            #now auc image
            img = auc_img
            vals = img[whole_mask]
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            auc_roi_avgs[0].append(mean)  
            auc_roi_avgs[1].append(std)

            #now auc image
            img = auc_l_img
            vals = img[whole_mask]
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            auc_l_roi_avgs[0].append(mean)  
            auc_l_roi_avgs[1].append(std)

            #now auc image
            img = auc_m_img
            vals = img[whole_mask]
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            auc_m_roi_avgs[0].append(mean)  
            auc_m_roi_avgs[1].append(std)

            #now auc image
            img = auc_h_img
            vals = img[whole_mask]
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            auc_h_roi_avgs[0].append(mean)  
            auc_h_roi_avgs[1].append(std)

            #now auc image
            img = auc_s0_img
            vals = img[whole_mask]
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            auc_s0_roi_avgs[0].append(mean)  
            auc_s0_roi_avgs[1].append(std)

            #now auc image
            img = auc_s0_l_img
            vals = img[whole_mask]
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            auc_s0_l_roi_avgs[0].append(mean)  
            auc_s0_l_roi_avgs[1].append(std)

            #now auc image
            img = auc_s0_m_img
            vals = img[whole_mask]
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            auc_s0_m_roi_avgs[0].append(mean)  
            auc_s0_m_roi_avgs[1].append(std)

            #now auc image
            img = auc_s0_h_img
            vals = img[whole_mask]
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            auc_s0_h_roi_avgs[0].append(mean)  
            auc_s0_h_roi_avgs[1].append(std)

            #now ADC image
            img = adc_img[0,:,:,:] #has extra index for S0 prediction
            vals = img[whole_mask]  
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            adc_roi_avgs[0].append(mean)   
            adc_roi_avgs[1].append(std)

           
            for seg in ["si", "ap", "ml"]:

                masks = getattr(mask_dict[roi], f"subseg_masks_deconv_{seg}")
                for mask in masks:
                    mask = mask.astype(bool)
                    #get biexp params:
                    for i in range(3):
                        img = biexp_img[i,...]
                        vals = img[mask]
                        mean = np.nanmean(vals)
                        std = np.nanstd(vals)
                        biexp_roi_avgs[i][0].append(mean)   #add all 3 biexp params
                        biexp_roi_avgs[i][1].append(std)

                    #get triexp params:
                    for i in range(5):
                        img = triexp_img[i,...]
                        vals = img[mask]
                        mean = np.nanmean(vals)
                        std = np.nanstd(vals)
                        triexp_roi_avgs[i][0].append(mean)    #add all 5 triexp params
                        triexp_roi_avgs[i][1].append(std)

                    #now auc image
                    img = auc_img
                    vals = img[mask]
                    mean = np.nanmean(vals)
                    std = np.nanstd(vals)
                    auc_roi_avgs[0].append(mean)  
                    auc_roi_avgs[1].append(std)

                    #now auc image
                    img = auc_l_img
                    vals = img[mask]
                    mean = np.nanmean(vals)
                    std = np.nanstd(vals)
                    auc_l_roi_avgs[0].append(mean)  
                    auc_l_roi_avgs[1].append(std)

                    #now auc image
                    img = auc_m_img
                    vals = img[mask]
                    mean = np.nanmean(vals)
                    std = np.nanstd(vals)
                    auc_m_roi_avgs[0].append(mean)  
                    auc_m_roi_avgs[1].append(std)

                    #now auc image
                    img = auc_h_img
                    vals = img[mask]
                    mean = np.nanmean(vals)
                    std = np.nanstd(vals)
                    auc_h_roi_avgs[0].append(mean)  
                    auc_h_roi_avgs[1].append(std)

                    #now auc image
                    img = auc_s0_img
                    vals = img[mask]
                    mean = np.nanmean(vals)
                    std = np.nanstd(vals)
                    auc_s0_roi_avgs[0].append(mean)  
                    auc_s0_roi_avgs[1].append(std)

                    #now auc image
                    img = auc_s0_l_img
                    vals = img[mask]
                    mean = np.nanmean(vals)
                    std = np.nanstd(vals)
                    auc_s0_l_roi_avgs[0].append(mean)  
                    auc_s0_l_roi_avgs[1].append(std)

                    #now auc image
                    img = auc_s0_m_img
                    vals = img[mask]
                    mean = np.nanmean(vals)
                    std = np.nanstd(vals)
                    auc_s0_m_roi_avgs[0].append(mean)  
                    auc_s0_m_roi_avgs[1].append(std)

                    #now auc image
                    img = auc_s0_h_img
                    vals = img[mask]
                    mean = np.nanmean(vals)
                    std = np.nanstd(vals)
                    auc_s0_h_roi_avgs[0].append(mean)  
                    auc_s0_h_roi_avgs[1].append(std)

                    #now ADC image
                    img = adc_img[0,:,:,:]
                    vals = img[mask]
                    mean = np.nanmean(vals)
                    std = np.nanstd(vals)
                    adc_roi_avgs[0].append(mean)   
                    adc_roi_avgs[1].append(std)

            all_stats[patient_num][roi]["adcs"] = adc_roi_avgs
            all_stats[patient_num][roi]["aucs"] = auc_roi_avgs
            all_stats[patient_num][roi]["aucs_l"] = auc_l_roi_avgs
            all_stats[patient_num][roi]["aucs_m"] = auc_m_roi_avgs
            all_stats[patient_num][roi]["aucs_h"] = auc_h_roi_avgs
            all_stats[patient_num][roi]["aucs_s0"] = auc_s0_roi_avgs
            all_stats[patient_num][roi]["aucs_s0_l"] = auc_s0_l_roi_avgs
            all_stats[patient_num][roi]["aucs_s0_m"] = auc_s0_m_roi_avgs
            all_stats[patient_num][roi]["aucs_s0_h"] = auc_s0_h_roi_avgs
            all_stats[patient_num][roi]["biexps"] = biexp_roi_avgs
            all_stats[patient_num][roi]["triexp"] = triexp_roi_avgs

        # with open(img_series_path, "wb") as fp:
        #     pickle.dump(img_dict_full, fp)

    all_biexp_avgs = [[[],[]],[[],[]],[[],[]]] 
    all_triexp_avgs = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]
    all_auc_avgs = [[],[]]
    all_auc_l_avgs = [[],[]]
    all_auc_m_avgs = [[],[]]
    all_auc_h_avgs = [[],[]]
    all_auc_s0_avgs = [[],[]]
    all_auc_s0_l_avgs = [[],[]]
    all_auc_s0_m_avgs = [[],[]]
    all_auc_s0_h_avgs = [[],[]]
    all_adc_avgs = [[],[]]
    for i in range(10):   #whole and 9 subsegs
        for j in range(3): #biexp
            vals = [] #for holding values from all patient/rois
            for patient in patient_nums:
                for roi in ["r_par", "l_par"]:
                    try:
                        vals.append(all_stats[patient][roi]["biexps"][j][0][i])
                    except:
                        continue
            avg = np.nanmean(vals)
            std = np.nanstd(vals)
            all_biexp_avgs[j][0].append(avg)
            all_biexp_avgs[j][1].append(std)

        for j in range(5): #triexp
            vals = [] #for holding values from all patient/rois
            for patient in patient_nums:
                for roi in ["r_par", "l_par"]:
                    try:
                        vals.append(all_stats[patient][roi]["triexp"][j][0][i])
                    except:
                        continue
            avg = np.nanmean(vals)
            std = np.nanstd(vals)
            all_triexp_avgs[j][0].append(avg)
            all_triexp_avgs[j][1].append(std)

        vals = [] #for holding values from all patient/rois
        for patient in patient_nums:
            for roi in ["r_par", "l_par"]:
                try:
                    vals.append(all_stats[patient][roi]["aucs"][0][i])
                except:
                    continue
        avg = np.nanmean(vals)
        std = np.nanstd(vals)
        all_auc_avgs[0].append(avg)
        all_auc_avgs[1].append(std)

        vals = [] #for holding values from all patient/rois
        for patient in patient_nums:
            for roi in ["r_par", "l_par"]:
                try:
                    vals.append(all_stats[patient][roi]["aucs_l"][0][i])
                except:
                    continue
        avg = np.nanmean(vals)
        std = np.nanstd(vals)
        all_auc_l_avgs[0].append(avg)
        all_auc_l_avgs[1].append(std)

        vals = [] #for holding values from all patient/rois
        for patient in patient_nums:
            for roi in ["r_par", "l_par"]:
                try:
                    vals.append(all_stats[patient][roi]["aucs_m"][0][i])
                except:
                    continue
        avg = np.nanmean(vals)
        std = np.nanstd(vals)
        all_auc_m_avgs[0].append(avg)
        all_auc_m_avgs[1].append(std)

        vals = [] #for holding values from all patient/rois
        for patient in patient_nums:
            for roi in ["r_par", "l_par"]:
                try:
                    vals.append(all_stats[patient][roi]["aucs_h"][0][i])
                except:
                    continue
        avg = np.nanmean(vals)
        std = np.nanstd(vals)
        all_auc_h_avgs[0].append(avg)
        all_auc_h_avgs[1].append(std)

        vals = [] #for holding values from all patient/rois
        for patient in patient_nums:
            for roi in ["r_par", "l_par"]:
                try:
                    vals.append(all_stats[patient][roi]["aucs_s0"][0][i])
                except:
                    continue
        avg = np.nanmean(vals)
        std = np.nanstd(vals)
        all_auc_s0_avgs[0].append(avg)
        all_auc_s0_avgs[1].append(std)

        vals = [] #for holding values from all patient/rois
        for patient in patient_nums:
            for roi in ["r_par", "l_par"]:
                try:
                    vals.append(all_stats[patient][roi]["aucs_s0_l"][0][i])
                except:
                    continue
        avg = np.nanmean(vals)
        std = np.nanstd(vals)
        all_auc_s0_l_avgs[0].append(avg)
        all_auc_s0_l_avgs[1].append(std)

        vals = [] #for holding values from all patient/rois
        for patient in patient_nums:
            for roi in ["r_par", "l_par"]:
                try:
                    vals.append(all_stats[patient][roi]["aucs_s0_m"][0][i])
                except:
                    continue
        avg = np.nanmean(vals)
        std = np.nanstd(vals)
        all_auc_s0_m_avgs[0].append(avg)
        all_auc_s0_m_avgs[1].append(std)

        vals = [] #for holding values from all patient/rois
        for patient in patient_nums:
            for roi in ["r_par", "l_par"]:
                try:
                    vals.append(all_stats[patient][roi]["aucs_s0_h"][0][i])
                except:
                    continue
        avg = np.nanmean(vals)
        std = np.nanstd(vals)
        all_auc_s0_h_avgs[0].append(avg)
        all_auc_s0_h_avgs[1].append(std)


        vals = [] #for holding values from all patient/rois
        for patient in patient_nums:
            for roi in ["r_par", "l_par"]:
                try:
                    vals.append(all_stats[patient][roi]["adcs"][0][i])
                except:
                    continue
        avg = np.nanmean(vals)
        std = np.nanstd(vals)
        all_adc_avgs[0].append(avg)
        all_adc_avgs[1].append(std)


    all_stats["all"] = {}
    all_stats["all"]["biexp"] = all_biexp_avgs
    all_stats["all"]["triexp"] = all_triexp_avgs
    all_stats["all"]["adc"] = all_adc_avgs
    all_stats["all"]["auc"] = all_auc_avgs
    all_stats["all"]["auc_l"] = all_auc_l_avgs
    all_stats["all"]["auc_m"] = all_auc_m_avgs
    all_stats["all"]["auc_h"] = all_auc_h_avgs
    all_stats["all"]["auc_s0"] = all_auc_s0_avgs
    all_stats["all"]["auc_s0_l"] = all_auc_s0_l_avgs
    all_stats["all"]["auc_s0_m"] = all_auc_s0_m_avgs
    all_stats["all"]["auc_s0_h"] = all_auc_s0_h_avgs
    

    save_path = os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", str("param_roi_stats_" + scan_type + "_" + str(deblurred) + ".txt"))
    with open(save_path, "wb") as fp:
        pickle.dump(all_stats, fp)
    return all_stats, dose_stats


def get_all_parotid_histogram_stats(deblurred=True, smoothed=False, scan_type="pre"):
    
    #dose stats already obtained 
    path = "/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep"
    #need to get all the different params for each different img type

    all_stats = {}

    #triexp order D, Dp1, Dp2, f1, f2
    #biexp order pD, D, f

    if scan_type == "pre":
        patient_nums = ["10", "11","12", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']
    else:
        patient_nums = ["10","11","12","13", "16"]

    for patient_num in patient_nums:
        
        all_stats[patient_num] = {}



        img_series_path = os.path.join(os.getcwd(), "data_mri")
        img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + scan_type + "diffusion_imgs"))
        with open(img_series_path, "rb") as fp:
            img_dict_full = pickle.load(fp)
        img_dict = img_dict_full["diffusion"]

        

        biexp_cord_hists = [[],[],[]] #for cord, will be whole gland values only
        triexp_cord_hists = [[],[],[],[],[]]
        auc_cord_hists = []
        curvature_cord_hists = []
        c20_cord_hists = []
        c50_cord_hists = []
        c100_cord_hists = []
        adc_cord_hists = []

        with open(os.path.join(os.path.join(os.getcwd(), "data_mri"), str("sivim" + patient_num + "_" + scan_type + "_MR_mask_dict")), "rb") as fp:
            mask_dict = pickle.load(fp)
        
       

        roi_masks = {}
        for roi in ["cord", "l_par", "r_par"]:
            if roi not in mask_dict.keys():
                continue
            all_stats[patient_num][roi] = {}
            #load different parameter images
            
            
            if "cord" not in roi.lower():
                all_stats[patient_num][roi] = {}
                #subsegment as needed
                Chopper.organ_chopper(mask_dict[roi], [0,0,2], name="segmented_contours_si")
                Chopper.organ_chopper(mask_dict[roi], [0,2,0], name="segmented_contours_ap")
                Chopper.organ_chopper(mask_dict[roi], [2,0,0], name="segmented_contours_ml")
                if roi == "r_par":
                    mask_dict[roi].segmented_contours_ml.reverse()   #sort medial to lateral

                coords_array = getattr(img_dict, f"deconv_coords_{roi}")
                #plot_3d_image(mask_dict[roi].mask_deconv)
                get_contour_masks.get_deconv_structure_masks(coords_array, mask_dict[roi])
            else:

                try:
                    whole_mask = mask_dict[roi].mask_deconv.astype(bool)
                    whole_mask = mask_dict[roi].mask_deconv.astype(bool)
                    auc_img = getattr(img_dict, f"auc_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    adc_img = getattr(img_dict, f"adc_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    curve_img = getattr(img_dict, f"curve_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    c20_img = getattr(img_dict, f"c20_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    c50_img = getattr(img_dict, f"c50_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    c100_img = getattr(img_dict, f"c100_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    triexp_img = getattr(img_dict, f"triexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    biexp_img = getattr(img_dict, f"biexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                except:
                    continue
                #get biexp params:
                bin_bounds = [[2e-3,0.1], [1e-4, 2e-3], [0, 0.4]]
                for i in range(3):
                    # plot_3d_image(biexp_img[i,...])
                    # plot_3d_image(whole_mask)
                    bins = np.linspace(bin_bounds[i][0], bin_bounds[i][1], 11)
                    img = biexp_img[i,...]
                    vals = img[whole_mask]
                    vals = vals[~np.isnan(vals)]
                    hist, bins = np.histogram(vals, bins=bins)

                    biexp_cord_hists[i].append([hist, bins])   #add all 3 biexp params



                all_stats[patient_num][roi]["biexps"] = biexp_cord_hists

                

                bin_bounds = [[1e-4,3e-3], [4e-3, 3e-2], [3e-2, 0.5], [0,0.4], [0,0.4]]
                #get triexp params:
                for i in range(5):
                    bins = np.linspace(bin_bounds[i][0], bin_bounds[i][1], 11)

                    img = triexp_img[i,...]
                    vals = img[whole_mask]
                    vals = img[whole_mask]
                    vals = vals[~np.isnan(vals)]
                    hist, bins = np.histogram(vals, bins=bins)
                    triexp_cord_hists[i].append([hist, bins])   #add all 5 triexp params

                all_stats[patient_num][roi]["triexp"] = triexp_cord_hists


            #now auc image
                bin_bounds = [300, 1000]
                bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
                img = auc_img
                vals = img[whole_mask]
                vals = img[whole_mask]
                vals = vals[~np.isnan(vals)]
                hist, bins = np.histogram(vals, bins=bin_bounds)

                auc_cord_hists.append([hist, bins])   
                all_stats[patient_num][roi]["aucs"] = auc_cord_hists


            #now ADC image
                bin_bounds = [1e-4, 4e-3]
                bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
                img = adc_img
                vals = img[whole_mask]
                vals = img[whole_mask]
                vals = vals[~np.isnan(vals)]
                hist, bins = np.histogram(vals, bins=bins)

                adc_cord_hists.append([hist, bins])  
                all_stats[patient_num][roi]["adcs"] = adc_cord_hists


            #now curvature image
                bin_bounds = [0,200]
                bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
                img = curve_img
                vals = img[whole_mask]
                vals = img[whole_mask]
                vals = vals[~np.isnan(vals)]
                hist, bins = np.histogram(vals, bins=bins)
                curvature_cord_hists.append([hist, bins])   
                all_stats[patient_num][roi]["curvatures"] = curvature_cord_hists

                bin_bounds = [0, 0.4]
                bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
                img = c20_img
                vals = img[whole_mask]
                vals = img[whole_mask]
                vals = vals[~np.isnan(vals)]
                hist, bins = np.histogram(vals, bins=bins)
                c20_cord_hists.append([hist, bins])   
                all_stats[patient_num][roi]["c20s"] = c20_cord_hists

                img = c50_img
                bin_bounds = [0, 0.4]
                bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
                vals = img[whole_mask]
                vals = img[whole_mask]
                vals = vals[~np.isnan(vals)]
                hist, bins = np.histogram(vals, bins=bins)
                c50_cord_hists.append([hist, bins])   
                all_stats[patient_num][roi]["c50s"] = c50_cord_hists

                img = c100_img
                bin_bounds = [0, 0.4]
                bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
                vals = img[whole_mask]
                vals = img[whole_mask]
                vals = vals[~np.isnan(vals)]
                hist, bins = np.histogram(vals, bins=bins)
                c100_cord_hists.append([hist, bins])   
                all_stats[patient_num][roi]["c100s"] = c100_cord_hists

                
                
                

        for roi in ["l_par", "r_par"]:
      

            try:
                whole_mask = mask_dict[roi].mask_deconv.astype(bool)
                auc_img = getattr(img_dict, f"auc_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                auc_l_img = getattr(img_dict, f"auc_l_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                auc_m_img = getattr(img_dict, f"auc_m_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                auc_h_img = getattr(img_dict, f"auc_h_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                auc_discrete_img = getattr(img_dict, f"auc_discrete_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                adc_img = getattr(img_dict, f"adc_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                curve_img = getattr(img_dict, f"curve_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                c20_img = getattr(img_dict, f"c20_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                c50_img = getattr(img_dict, f"c50_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                c100_img = getattr(img_dict, f"c100_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                triexp_img = getattr(img_dict, f"triexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                biexp_img = getattr(img_dict, f"biexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")

                auc_img_orig = getattr(img_dict, f"auc_img_{roi}_{False}_{smoothed}_{scan_type}")
                auc_l_img_orig = getattr(img_dict, f"auc_img_{roi}_{False}_{smoothed}_{scan_type}")
                auc_m_img_orig = getattr(img_dict, f"auc_img_{roi}_{False}_{smoothed}_{scan_type}")
                auc_h_img_orig = getattr(img_dict, f"auc_img_{roi}_{False}_{smoothed}_{scan_type}")
                auc_discrete_img_orig = getattr(img_dict, f"auc_img_{roi}_{False}_{smoothed}_{scan_type}")
                adc_img_orig = getattr(img_dict, f"adc_img_{roi}_{False}_{smoothed}_{scan_type}")
                curve_img_orig = getattr(img_dict, f"curve_img_{roi}_{False}_{smoothed}_{scan_type}")
                c20_img_orig = getattr(img_dict, f"c20_img_{roi}_{False}_{smoothed}_{scan_type}")
                c50_img_orig = getattr(img_dict, f"c50_img_{roi}_{False}_{smoothed}_{scan_type}")
                c100_img_orig = getattr(img_dict, f"c100_img_{roi}_{False}_{smoothed}_{scan_type}")
                triexp_img_orig = getattr(img_dict, f"triexp_img_{roi}_{False}_{smoothed}_{scan_type}")
                biexp_img_orig = getattr(img_dict, f"biexp_img_{roi}_{False}_{smoothed}_{scan_type}")

                auc_img[auc_img_orig == np.nan] = np.nan
                auc_l_img[auc_img_orig == np.nan] = np.nan
                auc_m_img[auc_img_orig == np.nan] = np.nan
                auc_h_img[auc_img_orig == np.nan] = np.nan
                auc_discrete_img[auc_img_orig == np.nan] = np.nan
                adc_img[auc_img_orig == np.nan] = np.nan
                curve_img[auc_img_orig == np.nan] = np.nan 
                c20_img[auc_img_orig == np.nan] = np.nan
                c50_img[auc_img_orig == np.nan] = np.nan
                c100_img[auc_img_orig == np.nan] = np.nan
                triexp_img[triexp_img_orig == np.nan] = np.nan
                biexp_img[biexp_img_orig == np.nan] = np.nan
            except:
                continue

            biexp_roi_hists = [[],[],[]] #for cord, will be whole gland values only
            triexp_roi_hists = [[],[],[],[],[]]
            auc_roi_hists = []
            auc_l_roi_hists = []
            auc_m_roi_hists = []
            auc_h_roi_hists = []
            auc_discrete_roi_hists = []
            curvature_roi_hists = []
            c20_roi_hists = []
            c50_roi_hists = []
            c100_roi_hists = []
            adc_roi_hists = []

            #first add whole gland stats
            whole_mask = mask_dict[roi].mask_deconv.astype(bool)
            #get biexp params:
            bin_bounds = [[2e-3,0.1], [1e-4, 2e-3], [0, 0.4]]
            for i in range(3):
                bins = np.linspace(bin_bounds[i][0], bin_bounds[i][1], 11)
                img = biexp_img[i,...]
                vals = img[whole_mask]
                vals = vals[~np.isnan(vals)]
                hist, bins = np.histogram(vals, bins=bins)
                biexp_roi_hists[i].append([hist, bins])   #add all 3 biexp params

            #get triexp params:
            bin_bounds = [[2e-4,2e-3], [4e-3, 3e-2], [3e-2, 0.5], [0,0.4], [0,0.4]]
            for i in range(5):
                bins = np.linspace(bin_bounds[i][0], bin_bounds[i][1], 11)
                img = triexp_img[i,...]
                vals = img[whole_mask]
                vals = vals[~np.isnan(vals)]
                hist, bins = np.histogram(vals, bins=bins)
                triexp_roi_hists[i].append([hist, bins]) 

            #now auc image
            bin_bounds = [300, 1000]
            bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
            img = auc_img
            vals = img[whole_mask]
            vals = vals[~np.isnan(vals)]
            hist, bins = np.histogram(vals, bins=bins)
            auc_roi_hists.append([hist, bins])  

            #now auc image
            bin_bounds = [300, 1000]
            bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
            img = auc_l_img
            vals = img[whole_mask]
            vals = vals[~np.isnan(vals)]
            hist, bins = np.histogram(vals, bins=10)
            auc_l_roi_hists.append([hist, bins])  

            #now auc image
            bin_bounds = [300, 1000]
            bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
            img = auc_m_img
            vals = img[whole_mask]
            vals = vals[~np.isnan(vals)]
            hist, bins = np.histogram(vals, bins=10)
            auc_m_roi_hists.append([hist, bins])  

            #now auc image
            bin_bounds = [300, 1000]
            bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
            img = auc_h_img
            vals = img[whole_mask]
            vals = vals[~np.isnan(vals)]
            hist, bins = np.histogram(vals, bins=10)
            auc_h_roi_hists.append([hist, bins])  

            #now auc image
            bin_bounds = [300, 1000]
            bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
            img = auc_discrete_img
            vals = img[whole_mask]
            vals = vals[~np.isnan(vals)]
            hist, bins = np.histogram(vals, bins=10)
            auc_discrete_roi_hists.append([hist, bins])  

            #now ADC image
            bin_bounds = [1e-4, 4e-3]
            bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
            img = adc_img
            vals = img[whole_mask]
            vals = vals[~np.isnan(vals)]
            hist, bins = np.histogram(vals, bins=bins)
            adc_roi_hists.append([hist, bins])  

            #now curvature image
            bin_bounds = [0,200]
            bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
            img = curve_img
            vals = img[whole_mask]
            vals = vals[~np.isnan(vals)]
            hist, bins = np.histogram(vals, bins=bins)
            curvature_roi_hists.append([hist, bins]) 

            bin_bounds = [0,0.4]
            bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
            img = c20_img
            vals = img[whole_mask]
            vals = vals[~np.isnan(vals)]
            hist, bins = np.histogram(vals, bins=bins)
            c20_roi_hists.append([hist, bins])   

            bin_bounds = [0,0.4]
            bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
            img = c50_img
            vals = img[whole_mask]
            vals = vals[~np.isnan(vals)]
            hist, bins = np.histogram(vals, bins=bins)
            c50_roi_hists.append([hist, bins])   

            bin_bounds = [0,0.4]
            bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
            img = c100_img
            vals = img[whole_mask]
            vals = vals[~np.isnan(vals)]
            hist, bins = np.histogram(vals, bins=bins)
            c100_roi_hists.append([hist, bins])   


            for seg in ["si", "ap", "ml"]:

                masks = getattr(mask_dict[roi], f"subseg_masks_deconv_{seg}")
                for mask in masks:
                    mask = mask.astype(bool)
                    #get biexp params:
                    bin_bounds = [[2e-3,0.1], [1e-4, 2e-3], [0, 0.4]]
                    for i in range(3):
                        bins = np.linspace(bin_bounds[i][0], bin_bounds[i][1], 11)
                        img = biexp_img[i,...]
                        vals = img[mask]
                        vals = vals[~np.isnan(vals)]
                        hist, bins = np.histogram(vals, bins=bins)
                        biexp_roi_hists[i].append([hist, bins])   #add all 3 biexp params

                    #get triexp params:
                    bin_bounds = [[2e-4,2e-3], [4e-3, 3e-2], [3e-2, 0.5], [0,0.4], [0,0.4]]
                    for i in range(5):
                        bins = np.linspace(bin_bounds[i][0], bin_bounds[i][1], 11)
                        img = triexp_img[i,...]
                        vals = img[mask]
                        vals = vals[~np.isnan(vals)]
                        hist, bins = np.histogram(vals, bins=bins)
                        triexp_roi_hists[i].append([hist, bins])   #add all 3 biexp params

                    #now auc image
                    bin_bounds = [300, 1000]
                    bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
                    img = auc_img
                    vals = img[mask]
                    vals = vals[~np.isnan(vals)]
                    hist, bins = np.histogram(vals, bins=bins)
                    auc_roi_hists.append([hist, bins])  

                            #now auc image
                    bin_bounds = [300, 1000]
                    bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
                    img = auc_l_img
                    vals = img[mask]
                    vals = vals[~np.isnan(vals)]
                    hist, bins = np.histogram(vals, bins=10)
                    auc_l_roi_hists.append([hist, bins])  

                    #now auc image
                    bin_bounds = [300, 1000]
                    bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
                    img = auc_m_img
                    vals = img[mask]
                    vals = vals[~np.isnan(vals)]
                    hist, bins = np.histogram(vals, bins=10)
                    auc_m_roi_hists.append([hist, bins])  

                    #now auc image
                    bin_bounds = [300, 1000]
                    bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
                    img = auc_h_img
                    vals = img[mask]
                    vals = vals[~np.isnan(vals)]
                    hist, bins = np.histogram(vals, bins=10)
                    auc_h_roi_hists.append([hist, bins])  

                    #now auc image
                    bin_bounds = [300, 1000]
                    bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
                    img = auc_discrete_img
                    vals = img[mask]
                    vals = vals[~np.isnan(vals)]
                    hist, bins = np.histogram(vals, bins=10)
                    auc_discrete_roi_hists.append([hist, bins])  

                    #now ADC image
                    bin_bounds = [1e-4, 4e-3]
                    bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
                    img = adc_img
                    vals = img[mask]
                    vals = vals[~np.isnan(vals)]
                    hist, bins = np.histogram(vals, bins=bins)
                    adc_roi_hists.append([hist, bins])  

                    #now curvature image
                    bin_bounds = [0,200]
                    bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
                    img = curve_img
                    vals = img[mask]
                    vals = vals[~np.isnan(vals)]
                    hist, bins = np.histogram(vals, bins=bins)
                    curvature_roi_hists.append([hist, bins]) 

                    bin_bounds = [0,0.4]
                    bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
                    img = c20_img
                    vals = img[mask]
                    vals = vals[~np.isnan(vals)]
                    hist, bins = np.histogram(vals, bins=bins)
                    c20_roi_hists.append([hist, bins])   

                    bin_bounds = [0,0.4]
                    bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
                    img = c50_img
                    vals = img[mask]
                    vals = vals[~np.isnan(vals)]
                    hist, bins = np.histogram(vals, bins=bins)
                    c50_roi_hists.append([hist, bins])   

                    bin_bounds = [0,0.4]
                    bins = np.linspace(bin_bounds[0], bin_bounds[1], 11)
                    img = c100_img
                    vals = img[mask]
                    vals = vals[~np.isnan(vals)]
                    hist, bins = np.histogram(vals, bins=bins)
                    c100_roi_hists.append([hist, bins])   



            all_stats[patient_num][roi]["curvatures"] = curvature_roi_hists
            all_stats[patient_num][roi]["c20s"] = c20_roi_hists
            all_stats[patient_num][roi]["c50s"] = c50_roi_hists
            all_stats[patient_num][roi]["c100s"] = c100_roi_hists
            all_stats[patient_num][roi]["adcs"] = adc_roi_hists
            all_stats[patient_num][roi]["aucs"] = auc_roi_hists
            all_stats[patient_num][roi]["aucs_l"] = auc_l_roi_hists
            all_stats[patient_num][roi]["aucs_m"] = auc_m_roi_hists
            all_stats[patient_num][roi]["aucs_h"] = auc_h_roi_hists
            all_stats[patient_num][roi]["aucs_discrete"] = auc_discrete_roi_hists
            all_stats[patient_num][roi]["biexps"] = biexp_roi_hists
            all_stats[patient_num][roi]["triexp"] = triexp_roi_hists

    
    #

    save_path = os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", str("param_roi_hists_" + scan_type + "_" + str(deblurred) + ".txt"))
    with open(save_path, "wb") as fp:
        pickle.dump(all_stats, fp)

def dose_to_ivim_change_analysis_rel(deblurred=True):
    #relative ivim parameter change version, as opposed to absolute
    #first need to run get_all_parotid_param_and_dose_stats. dose stats were calculated previously in cross modality code.

    #load dose stats 
    print(deblurred)
    path = "/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep"
    with open(os.path.join(path, "dose_data.txt"), "rb") as fp:
        dose_stats = pickle.load(fp)
    #load ivim stats 
    save_path = os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", str("param_roi_stats_" + "pre" + "_" + str(deblurred) + ".txt"))
    with open(save_path, "rb") as fp:
        ivim_stats_pre = pickle.load(fp)

    save_path = os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", str("param_roi_stats_" + "post" + "_" + str(deblurred) + ".txt"))
    with open(save_path, "rb") as fp:
        ivim_stats_post = pickle.load(fp)
    
    biexp_stats = ivim_stats_pre["all"]["biexp"]
    triexp_stats = ivim_stats_pre["all"]["triexp"]
    adc_stats = ivim_stats_pre["all"]["adc"]
    auc_stats = ivim_stats_pre["all"]["auc"]
    auc_l_stats = ivim_stats_pre["all"]["auc_l"]
    auc_m_stats = ivim_stats_pre["all"]["auc_m"]
    auc_h_stats = ivim_stats_pre["all"]["auc_h"]
    auc_s0_stats = ivim_stats_pre["all"]["auc_s0"]
    auc_s0_l_stats = ivim_stats_pre["all"]["auc_s0_l"]
    auc_s0_m_stats = ivim_stats_pre["all"]["auc_s0_m"]
    auc_s0_h_stats = ivim_stats_pre["all"]["auc_s0_h"]

    patient_nums = ["10","11","12","13", "16"]
    rois = ["l_par", "r_par"]


    #now want to get the predictive power of the whole mean dose for whole mean IVIM change first. 

    from sklearn.linear_model import LinearRegression

    whole_changes_biexp = [[],[],[]]
    whole_changes_triexp = [[],[],[],[],[]]
    whole_changes_adc = []
    whole_changes_curvature = []
    whole_changes_c20 = []
    whole_changes_c50 = []
    whole_changes_c100 = []
    whole_changes_auc = []
    whole_changes_auc_l = []
    whole_changes_auc_m = []
    whole_changes_auc_h = []
    whole_changes_auc_s0 = []
    whole_changes_auc_s0_l = []
    whole_changes_auc_s0_m = []
    whole_changes_auc_s0_h = []

    abs_whole_pre_biexp = [[],[],[]]
    abs_whole_pre_triexp = [[],[],[],[],[]]
    abs_whole_pre_adc = []
    abs_whole_pre_curvature = []
    abs_whole_pre_c20 = []
    abs_whole_pre_c50 = []
    abs_whole_pre_c100 = []
    abs_whole_pre_auc = []
    abs_whole_pre_auc_l = []
    abs_whole_pre_auc_m = []
    abs_whole_pre_auc_h = []
    abs_whole_pre_auc_s0 = []
    abs_whole_pre_auc_s0_l = []
    abs_whole_pre_auc_s0_m = []
    abs_whole_pre_auc_s0_h = []

    abs_whole_post_biexp = [[],[],[]]
    abs_whole_post_triexp = [[],[],[],[],[]]
    abs_whole_post_adc = []
    abs_whole_post_curvature = []
    abs_whole_post_c20 = []
    abs_whole_post_c50 = []
    abs_whole_post_c100 = []
    abs_whole_post_auc = []
    abs_whole_post_auc_l = []
    abs_whole_post_auc_m = []
    abs_whole_post_auc_h = []
    abs_whole_post_auc_s0 = []
    abs_whole_post_auc_s0_l = []
    abs_whole_post_auc_s0_m = []
    abs_whole_post_auc_s0_h = []

    whole_doses = []
    for patient in patient_nums:
        patient_dose_dict = "SIVIM" + str(patient)
        for roi in rois: 
            try:
                ivim_params_pre = ivim_stats_pre[patient][roi]
                ivim_params_post = ivim_stats_post[patient][roi]
            except:
                print(f"Failed for {roi} for {patient}")
                continue
            whole_doses.append(dose_stats[patient_dose_dict][roi]["whole"][0])
            
            whole_changes_adc.append((ivim_params_post["adcs"][0][0] - ivim_params_pre["adcs"][0][0])/ ivim_params_pre["adcs"][0][0])
            whole_changes_auc.append((ivim_params_post["aucs"][0][0] - ivim_params_pre["aucs"][0][0])/ ivim_params_pre["aucs"][0][0])
            whole_changes_auc_l.append((ivim_params_post["aucs_l"][0][0] - ivim_params_pre["aucs_l"][0][0])/ ivim_params_pre["aucs_l"][0][0])
            whole_changes_auc_m.append((ivim_params_post["aucs_m"][0][0] - ivim_params_pre["aucs_m"][0][0])/ ivim_params_pre["aucs_m"][0][0])
            whole_changes_auc_h.append((ivim_params_post["aucs_h"][0][0] - ivim_params_pre["aucs_h"][0][0])/ ivim_params_pre["aucs_h"][0][0])
            whole_changes_auc_s0.append((ivim_params_post["aucs_s0"][0][0] - ivim_params_pre["aucs_s0"][0][0])/ ivim_params_pre["aucs_s0"][0][0])
            whole_changes_auc_s0_l.append((ivim_params_post["aucs_s0_l"][0][0] - ivim_params_pre["aucs_s0_l"][0][0])/ ivim_params_pre["aucs_s0_l"][0][0])
            whole_changes_auc_s0_m.append((ivim_params_post["aucs_s0_m"][0][0] - ivim_params_pre["aucs_s0_m"][0][0])/ ivim_params_pre["aucs_s0_m"][0][0])
            whole_changes_auc_s0_h.append((ivim_params_post["aucs_s0_h"][0][0] - ivim_params_pre["aucs_s0_h"][0][0])/ ivim_params_pre["aucs_s0_h"][0][0])
            # whole_changes_curvature.append((ivim_params_post["curvatures"][0][0] - ivim_params_pre["curvatures"][0][0])/ ivim_params_pre["curvatures"][0][0])
            # whole_changes_c20.append((ivim_params_post["c20s"][0][0] - ivim_params_pre["c20s"][0][0])/ ivim_params_pre["c20s"][0][0])
            # whole_changes_c50.append((ivim_params_post["c50s"][0][0] - ivim_params_pre["c50s"][0][0])/ ivim_params_pre["c50s"][0][0])
            # whole_changes_c100.append((ivim_params_post["c100s"][0][0] - ivim_params_pre["c100s"][0][0])/ ivim_params_pre["c100s"][0][0])

            abs_whole_pre_adc.append((ivim_params_pre["adcs"][0][0]))
            abs_whole_pre_auc.append((ivim_params_pre["aucs"][0][0]))
            abs_whole_pre_auc_l.append((ivim_params_pre["aucs_l"][0][0]))
            abs_whole_pre_auc_m.append((ivim_params_pre["aucs_m"][0][0]))
            abs_whole_pre_auc_h.append((ivim_params_pre["aucs_h"][0][0]))
            abs_whole_pre_auc_s0.append((ivim_params_pre["aucs_s0"][0][0]))
            abs_whole_pre_auc_s0_l.append((ivim_params_pre["aucs_s0_l"][0][0]))
            abs_whole_pre_auc_s0_m.append((ivim_params_pre["aucs_s0_m"][0][0]))
            abs_whole_pre_auc_s0_h.append((ivim_params_pre["aucs_s0_h"][0][0]))
            # abs_whole_changes_curvature.append((ivim_params_post["curvatures"][0][0] - ivim_params_pre["curvatures"][0][0]))
            # abs_whole_changes_c20.append((ivim_params_post["c20s"][0][0] - ivim_params_pre["c20s"][0][0]))
            # abs_whole_changes_c50.append((ivim_params_post["c50s"][0][0] - ivim_params_pre["c50s"][0][0]))
            # abs_whole_changes_c100.append((ivim_params_post["c100s"][0][0] - ivim_params_pre["c100s"][0][0]))

            abs_whole_post_adc.append((ivim_params_post["adcs"][0][0]))
            abs_whole_post_auc.append((ivim_params_post["aucs"][0][0]))
            abs_whole_post_auc_l.append((ivim_params_post["aucs_l"][0][0]))
            abs_whole_post_auc_m.append((ivim_params_post["aucs_m"][0][0]))
            abs_whole_post_auc_h.append((ivim_params_post["aucs_h"][0][0]))
            abs_whole_post_auc_s0.append((ivim_params_post["aucs_s0"][0][0]))
            abs_whole_post_auc_s0_l.append((ivim_params_post["aucs_s0_l"][0][0]))
            abs_whole_post_auc_s0_m.append((ivim_params_post["aucs_s0_m"][0][0]))
            abs_whole_post_auc_s0_h.append((ivim_params_post["aucs_s0_h"][0][0]))
            # abs_whole_post_curvature.append((ivim_params_post["curvatures"][0][0]))
            # abs_whole_post_c20.append((ivim_params_post["c20s"][0][0]))
            # abs_whole_post_c50.append((ivim_params_post["c50s"][0][0]))
            # abs_whole_post_c100.append((ivim_params_post["c100s"][0][0]))
            for i in range(3):
                whole_changes_biexp[i].append((ivim_params_post["biexps"][i][0][0] -ivim_params_pre["biexps"][i][0][0])/ ivim_params_pre["biexps"][i][0][0])
                abs_whole_pre_biexp[i].append((ivim_params_pre["biexps"][i][0][0]))
                abs_whole_post_biexp[i].append((ivim_params_post["biexps"][i][0][0]))
            for i in range(5):
                whole_changes_triexp[i].append((ivim_params_post["triexp"][i][0][0]-ivim_params_pre["triexp"][i][0][0]) / ivim_params_pre["triexp"][i][0][0])
                abs_whole_pre_triexp[i].append((ivim_params_pre["triexp"][i][0][0]))
                abs_whole_post_triexp[i].append((ivim_params_post["triexp"][i][0][0]))

    whole_doses = np.array(whole_doses).reshape(-1,1)
    from scipy import stats
    model = LinearRegression()

    corr_adc, p_adc = spearmanr(whole_doses, whole_changes_adc)
    corr_adc = np.abs(corr_adc)
    X = np.column_stack((whole_doses, abs_whole_pre_adc))
    y = (np.array(abs_whole_post_adc) - adc_stats[0][0]) / adc_stats[1][0]
    model.fit(X, y)
    slope_dose_adc = model.coef_[0]
    score_adc = model.score(X,y)
    t_adc, p_adc = stats.ttest_rel(abs_whole_post_adc, abs_whole_pre_adc)


    corr_auc, p_auc = spearmanr(whole_doses, whole_changes_auc)
    corr_auc = np.abs(corr_auc)
    X = np.column_stack((whole_doses, abs_whole_pre_auc))
    y = (np.array(abs_whole_post_auc)- auc_stats[0][0]) / auc_stats[1][0]
    model = LinearRegression()
    model.fit(X, y)
    slope_dose_auc = model.coef_[0]
    score_auc = model.score(X,y)
    

    corr_auc_l, p_auc_l = spearmanr(whole_doses, whole_changes_auc_l)
    corr_auc_l = np.abs(corr_auc_l)
    X = np.column_stack((whole_doses, abs_whole_pre_auc_l))
    y = (np.array(abs_whole_post_auc_l)- auc_stats[0][0]) / auc_stats[1][0]
    model = LinearRegression()
    model.fit(X, y)
    slope_dose_auc_l = model.coef_[0]
    score_auc_l = model.score(X,y)


    corr_auc_m, p_auc_m = spearmanr(whole_doses, whole_changes_auc_m)
    corr_auc_m = np.abs(corr_auc_m)
    X = np.column_stack((whole_doses, abs_whole_pre_auc_m))
    y = (np.array(abs_whole_post_auc_m)  - auc_m_stats[0][0]) / auc_m_stats[1][0]
    model = LinearRegression()
    model.fit(X, y)
    slope_dose_auc_m = model.coef_[0]
    score_auc_m = model.score(X,y)

    corr_auc_h, p_auc_h = spearmanr(whole_doses, whole_changes_auc_h)
    corr_auc_h = np.abs(corr_auc_h)
    X = np.column_stack((whole_doses, abs_whole_pre_auc_h))
    y = (np.array(abs_whole_post_auc_h) - auc_h_stats[0][0]) / auc_h_stats[1][0]
    model = LinearRegression()
    model.fit(X, y)
    slope_dose_auc_h = model.coef_[0]
    score_auc_h = model.score(X,y)

    corr_auc_s0, p_auc_s0 = spearmanr(whole_doses, whole_changes_auc_s0)
    corr_auc_s0 = np.abs(corr_auc_s0)
    X = np.column_stack((whole_doses, abs_whole_pre_auc_s0))
    y = (np.array(abs_whole_post_auc_s0)- auc_s0_stats[0][0]) / auc_s0_stats[1][0]
    model = LinearRegression()
    model.fit(X, y)
    slope_dose_auc_s0 = model.coef_[0]
    score_auc_s0 = model.score(X,y)
    t_auc, p_auc = stats.ttest_rel(abs_whole_post_auc_s0, abs_whole_pre_auc_s0)

    corr_auc_s0_l, p_auc_s0_l = spearmanr(whole_doses, whole_changes_auc_s0_l)
    corr_auc_s0_l = np.abs(corr_auc_s0_l)
    X = np.column_stack((whole_doses, abs_whole_pre_auc_s0_l))
    y = (np.array(abs_whole_post_auc_s0_l) - auc_s0_l_stats[0][0]) / auc_s0_l_stats[1][0]
    model = LinearRegression()
    model.fit(X, y)
    slope_dose_auc_s0_l = model.coef_[0]
    score_auc_s0_l = model.score(X,y)
    t_auc_l, p_auc_l = stats.ttest_rel(abs_whole_post_auc_s0_l, abs_whole_pre_auc_s0_l)

    corr_auc_s0_m, p_auc_s0_m = spearmanr(whole_doses, whole_changes_auc_s0_m)
    corr_auc_s0_m = np.abs(corr_auc_s0_m)
    X = np.column_stack((whole_doses, abs_whole_pre_auc_s0_m))
    y = (np.array(abs_whole_post_auc_s0_m) - auc_s0_m_stats[0][0]) / auc_s0_m_stats[1][0]
    model = LinearRegression()
    model.fit(X, y)
    slope_dose_auc_s0_m = model.coef_[0]
    score_auc_s0_m = model.score(X,y)
    t_auc_m, p_auc_m = stats.ttest_rel(abs_whole_post_auc_s0_m, abs_whole_pre_auc_s0_m)

    corr_auc_s0_h, p_auc_s0_h = spearmanr(whole_doses, whole_changes_auc_s0_h)
    corr_auc_s0_h = np.abs(corr_auc_s0_h)

    X = np.column_stack((whole_doses, abs_whole_pre_auc_s0_h))
    y = (np.array(abs_whole_post_auc_s0_h) - auc_s0_h_stats[0][0]) / auc_s0_h_stats[1][0]
    model = LinearRegression()
    model.fit(X, y)
    slope_dose_auc_s0_h = model.coef_[0]
    t_auc_h, p_auc_h = stats.ttest_rel(abs_whole_post_auc_s0_h, abs_whole_pre_auc_s0_h)
    score_auc_s0_h = model.score(X,y)

    corr_biexp = []
    p_biexp = []
    corr_triexp = []
    p_triexp = []

    slope_dose_biexp = []
    slope_dose_triexp = []
    score_biexp = []
    score_triexp = []

    ts_biexp = []
    ps_biexp = []
    ts_triexp = []
    ps_triexp = []

    for i in range(3):
        X = np.column_stack((whole_doses, abs_whole_pre_biexp[i]))
        y = (np.array(abs_whole_post_biexp[i])- biexp_stats[i][0][0]) / biexp_stats[i][1][0]
        model = LinearRegression()
        model.fit(X, y)
        score_biexp.append(model.score(X,y))
        slope_dose_biexp.append(model.coef_[0])
        corr, p = spearmanr(whole_doses, whole_changes_biexp[i])
        corr_biexp.append(np.abs(corr))
        p_biexp.append(p)
        t, p = stats.ttest_rel(abs_whole_post_biexp[i], abs_whole_pre_biexp[i])
        ts_biexp.append(t)
        ps_biexp.append(p)

    for i in range(5):
        X = np.column_stack((whole_doses, abs_whole_pre_triexp[i]))
        y = (np.array(abs_whole_post_triexp[i])- triexp_stats[i][0][0]) / triexp_stats[i][1][0]
        model = LinearRegression()
        model.fit(X, y)
        score_triexp.append(model.score(X,y))
        slope_dose_triexp.append(model.coef_[0])
        corr, p = spearmanr(whole_doses, whole_changes_triexp[i])
        corr_triexp.append(np.abs(corr))
        p_triexp.append(p)
        t, p = stats.ttest_rel(abs_whole_post_triexp[i], abs_whole_pre_triexp[i])
        ts_triexp.append(t)
        ps_triexp.append(p)

    #make plot of r2 
    #triexp order  D.item(), Dp1.item(), Dp2.item(), f1.item(), f2.item()
    #biexp Dp D f
    fig, ax = plt.subplots(figsize=(20,20))


    # #do showing correlations
    params = ["$D_{biexp}$", "$D*_{biexp}$", "$f_{biexp}$", "$D_{triexp}$",  "$D*_1$", "$f_1$", "$D*_2$",  "$f_2$", "ADC", "AUC", "$AUC_l$", "$AUC_m$", "$AUC_h$"]
    
    all_scores = [corr_biexp[1], corr_biexp[0], corr_biexp[2], corr_triexp[0], corr_triexp[1], corr_triexp[2], corr_triexp[3], corr_triexp[4],corr_adc, corr_auc_s0, corr_auc_s0_l, corr_auc_s0_m, corr_auc_s0_h]
    cmap = plt.get_cmap('jet')
    normalize = plt.Normalize(vmin=min(all_scores), vmax=max(all_scores))
    colors = cmap(normalize(all_scores))
    ax.grid(alpha=0.5, zorder=0)
    ax.bar(params, all_scores, color="deepskyblue", edgecolor="dodgerblue", zorder=5)
    #ax.set_title("Whole Correlations")
    ax.set_ylabel("$r_s$", fontsize=24)
    ax.set_ylim([0,0.9])
    ax.yaxis.set_tick_params(labelsize=24)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    
    ax.set_xticklabels(params, rotation=45, ha='right', fontsize=24)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, orientation='vertical')
    # cbar.set_label('$r_s$', fontsize=16)
    plt.show()

    fig, ax = plt.subplots(figsize=(20,20))
    # # #do showing regression dose slope
    params = ["$D_{biexp}$", "$D*_{biexp}$", "$f_{biexp}$", "$D_{triexp}$",  "$D*_1$", "$f_1$","$D*_2$",  "$f_2$", "ADC", "AUC", "$AUC_l$", "$AUC_m$", "$AUC_h$"]
    
    all_scores = [slope_dose_biexp[1], slope_dose_biexp[0], slope_dose_biexp[2], slope_dose_triexp[0], slope_dose_triexp[1], slope_dose_triexp[2], slope_dose_triexp[3], slope_dose_triexp[4],slope_dose_adc, slope_dose_auc_s0, slope_dose_auc_s0_l, slope_dose_auc_s0_m, slope_dose_auc_s0_h]
    cmap = plt.get_cmap('jet')
    normalize = plt.Normalize(vmin=min(all_scores), vmax=max(all_scores))
    colors = cmap(normalize(all_scores))
    ax.grid(alpha=0.5, zorder=0)
    ax.bar(params, all_scores, color="mediumspringgreen", edgecolor="mediumseagreen", zorder=5)
    #ax.set_title("Whole Correlations")
    ax.set_ylabel(r"$\frac{\Delta \, Z_f}{\Delta \, D}$", fontsize=24)
    ax.set_ylim([-0.05,0.05])
    ax.yaxis.set_tick_params(labelsize=24)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    
    ax.set_xticklabels(params, rotation=45, ha='right', fontsize=24)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, orientation='vertical')
    # cbar.set_label('$r_s$', fontsize=16)
    plt.show()

    subseg_changes_biexp = [[],[],[]]
    subseg_changes_triexp = [[],[],[],[],[]]
    subseg_changes_adc = []
    subseg_changes_curvature = []
    subseg_changes_c20 = []
    subseg_changes_c50 = []
    subseg_changes_c100 = []
    subseg_changes_auc = []
    subseg_changes_auc_l = []
    subseg_changes_auc_m = []
    subseg_changes_auc_h = []
    subseg_changes_auc_s0 = []
    subseg_changes_auc_s0_l = []
    subseg_changes_auc_s0_m = []
    subseg_changes_auc_s0_h = []

    abs_subseg_pre_biexp = [[],[],[]]
    abs_subseg_pre_triexp = [[],[],[],[],[]]
    abs_subseg_pre_adc = []
    abs_subseg_pre_curvature = []
    abs_subseg_pre_c20 = []
    abs_subseg_pre_c50 = []
    abs_subseg_pre_c100 = []
    abs_subseg_pre_auc = []
    abs_subseg_pre_auc_l = []
    abs_subseg_pre_auc_m = []
    abs_subseg_pre_auc_h = []
    abs_subseg_pre_auc_s0 = []
    abs_subseg_pre_auc_s0_l = []
    abs_subseg_pre_auc_s0_m = []
    abs_subseg_pre_auc_s0_h = []

    abs_subseg_post_biexp = [[],[],[]]
    abs_subseg_post_triexp = [[],[],[],[],[]]
    abs_subseg_post_adc = []
    abs_subseg_post_curvature = []
    abs_subseg_post_c20 = []
    abs_subseg_post_c50 = []
    abs_subseg_post_c100 = []
    abs_subseg_post_auc = []
    abs_subseg_post_auc_l = []
    abs_subseg_post_auc_m = []
    abs_subseg_post_auc_h = []
    abs_subseg_post_auc_s0 = []
    abs_subseg_post_auc_s0_l = []
    abs_subseg_post_auc_s0_m = []
    abs_subseg_post_auc_s0_h = []

    subseg_doses = []

    for j in range(9):    #make separate lists in each, for all 9 subsegs
        subseg_changes_adc.append([])
        subseg_changes_curvature.append([])
        subseg_changes_c20.append([])
        subseg_changes_c50.append([])
        subseg_changes_c100.append([])
        subseg_changes_auc.append([])
        subseg_changes_auc_l.append([])
        subseg_changes_auc_m.append([])
        subseg_changes_auc_h.append([])
        subseg_changes_auc_s0.append([])
        subseg_changes_auc_s0_l.append([])
        subseg_changes_auc_s0_m.append([])
        subseg_changes_auc_s0_h.append([])

        abs_subseg_pre_adc.append([])
        abs_subseg_pre_auc.append([])
        abs_subseg_pre_auc_l.append([])
        abs_subseg_pre_auc_m.append([])
        abs_subseg_pre_auc_h.append([])
        abs_subseg_pre_auc_s0.append([])
        abs_subseg_pre_auc_s0_l.append([])
        abs_subseg_pre_auc_s0_m.append([])
        abs_subseg_pre_auc_s0_h.append([])
        #abs_subseg_pre_curvature[j-1].append((ivim_params_post["curvatures"][0][j] -ivim_params_pre["curvatures"][0][j])/ ivim_params_pre["curvatures"][0][j])
        # abs_subseg_pre_c20[j-1].append((ivim_params_post["c20s"][0][j] - ivim_params_pre["c20s"][0][j]) / ivim_params_pre["c20s"][0][j])
        # abs_subseg_pre_c50[j-1].append((ivim_params_post["c50s"][0][j] - ivim_params_pre["c50s"][0][j])/ ivim_params_pre["c50s"][0][j])
        # abs_subseg_pre_c100[j-1].append((ivim_params_post["c100s"][0][j] - ivim_params_pre["c100s"][0][j])/ ivim_params_pre["c100s"][0][j])

        abs_subseg_post_adc.append([])
        abs_subseg_post_auc.append([])
        abs_subseg_post_auc_l.append([])
        abs_subseg_post_auc_m.append([])
        abs_subseg_post_auc_h.append([])
        abs_subseg_post_auc_s0.append([])
        abs_subseg_post_auc_s0_l.append([])
        abs_subseg_post_auc_s0_m.append([])
        abs_subseg_post_auc_s0_h.append([])

        subseg_doses.append([])
        for jj in range(3):
            subseg_changes_biexp[jj].append([])
            abs_subseg_pre_biexp[jj].append([])
            abs_subseg_post_biexp[jj].append([])

        for jj in range(5):
            subseg_changes_triexp[jj].append([])
            abs_subseg_pre_triexp[jj].append([])
            abs_subseg_post_triexp[jj].append([])

    for patient in patient_nums:
        patient_dose_dict = "SIVIM" + str(patient)
        for roi in rois: 
            try:
                ivim_params_pre = ivim_stats_pre[patient][roi]
                ivim_params_post = ivim_stats_post[patient][roi]
                dose_stat = dose_stats[patient_dose_dict][roi]
            except:
                print(f"Failed for {roi} for {patient}")
                continue

            s_idx = 0
            for ss in ["si", "ap", "ml"]:
                for s in range(3):
                    subseg_doses[s_idx].append(dose_stat[ss][s][0])
                    s_idx += 1

            for j in range(1,10):  
                
                subseg_changes_adc[j-1].append((ivim_params_post["adcs"][0][j] -ivim_params_pre["adcs"][0][j])/ ivim_params_pre["adcs"][0][j])
                subseg_changes_auc[j-1].append((ivim_params_post["aucs"][0][j] -ivim_params_pre["aucs"][0][j])/ ivim_params_pre["aucs"][0][j])
                subseg_changes_auc_l[j-1].append((ivim_params_post["aucs_l"][0][j] - ivim_params_pre["aucs_l"][0][j])/ ivim_params_pre["aucs_l"][0][j])
                subseg_changes_auc_m[j-1].append((ivim_params_post["aucs_m"][0][j] - ivim_params_pre["aucs_m"][0][j])/ ivim_params_pre["aucs_m"][0][j])
                subseg_changes_auc_h[j-1].append((ivim_params_post["aucs_h"][0][j] - ivim_params_pre["aucs_h"][0][j])/ ivim_params_pre["aucs_h"][0][j])
                subseg_changes_auc_s0[j-1].append((ivim_params_post["aucs_s0"][0][j] - ivim_params_pre["aucs_s0"][0][j])/ ivim_params_pre["aucs_s0"][0][j])
                subseg_changes_auc_s0_l[j-1].append((ivim_params_post["aucs_s0_l"][0][j] - ivim_params_pre["aucs_s0_l"][0][j])/ ivim_params_pre["aucs_s0_l"][0][j])
                subseg_changes_auc_s0_m[j-1].append((ivim_params_post["aucs_s0_m"][0][j] - ivim_params_pre["aucs_s0_m"][0][j])/ ivim_params_pre["aucs_s0_m"][0][j])
                subseg_changes_auc_s0_h[j-1].append((ivim_params_post["aucs_s0_h"][0][j] - ivim_params_pre["aucs_s0_h"][0][j])/ ivim_params_pre["aucs_s0_h"][0][j])
                #subseg_changes_curvature[j-1].append((ivim_params_post["curvatures"][0][j] -ivim_params_pre["curvatures"][0][j])/ ivim_params_pre["curvatures"][0][j])
                # subseg_changes_c20[j-1].append((ivim_params_post["c20s"][0][j] - ivim_params_pre["c20s"][0][j]) / ivim_params_pre["c20s"][0][j])
                # subseg_changes_c50[j-1].append((ivim_params_post["c50s"][0][j] - ivim_params_pre["c50s"][0][j])/ ivim_params_pre["c50s"][0][j])
                # subseg_changes_c100[j-1].append((ivim_params_post["c100s"][0][j] - ivim_params_pre["c100s"][0][j])/ ivim_params_pre["c100s"][0][j])
                for i in range(3):
                    subseg_changes_biexp[i][j-1].append((ivim_params_post["biexps"][i][0][j] - ivim_params_pre["biexps"][i][0][j])/ ivim_params_pre["biexps"][i][0][j])
                for i in range(5):
                    subseg_changes_triexp[i][j-1].append((ivim_params_post["triexp"][i][0][j] - ivim_params_pre["triexp"][i][0][j])/ ivim_params_pre["triexp"][i][0][j])

                abs_subseg_pre_adc[j-1].append(ivim_params_pre["adcs"][0][j])
                abs_subseg_pre_auc[j-1].append(ivim_params_pre["aucs"][0][j])
                abs_subseg_pre_auc_l[j-1].append(ivim_params_pre["aucs_l"][0][j])
                abs_subseg_pre_auc_m[j-1].append(ivim_params_pre["aucs_m"][0][j])
                abs_subseg_pre_auc_h[j-1].append(ivim_params_pre["aucs_h"][0][j])
                abs_subseg_pre_auc_s0[j-1].append(ivim_params_pre["aucs_s0"][0][j])
                abs_subseg_pre_auc_s0_l[j-1].append(ivim_params_pre["aucs_s0_l"][0][j])
                abs_subseg_pre_auc_s0_m[j-1].append(ivim_params_pre["aucs_s0_m"][0][j])
                abs_subseg_pre_auc_s0_h[j-1].append(ivim_params_pre["aucs_s0_h"][0][j])
                #abs_subseg_pre_curvature[j-1].append((ivim_params_post["curvatures"][0][j] -ivim_params_pre["curvatures"][0][j])/ ivim_params_pre["curvatures"][0][j])
                # abs_subseg_pre_c20[j-1].append((ivim_params_post["c20s"][0][j] - ivim_params_pre["c20s"][0][j]) / ivim_params_pre["c20s"][0][j])
                # abs_subseg_pre_c50[j-1].append((ivim_params_post["c50s"][0][j] - ivim_params_pre["c50s"][0][j])/ ivim_params_pre["c50s"][0][j])
                # abs_subseg_pre_c100[j-1].append((ivim_params_post["c100s"][0][j] - ivim_params_pre["c100s"][0][j])/ ivim_params_pre["c100s"][0][j])
                for i in range(3):
                    abs_subseg_pre_biexp[i][j-1].append(ivim_params_pre["biexps"][i][0][j])
                for i in range(5):
                    abs_subseg_pre_triexp[i][j-1].append(ivim_params_pre["triexp"][i][0][j])

                abs_subseg_post_adc[j-1].append(ivim_params_post["adcs"][0][j])
                abs_subseg_post_auc[j-1].append(ivim_params_post["aucs"][0][j])
                abs_subseg_post_auc_l[j-1].append(ivim_params_post["aucs_l"][0][j])
                abs_subseg_post_auc_m[j-1].append(ivim_params_post["aucs_m"][0][j])
                abs_subseg_post_auc_h[j-1].append(ivim_params_post["aucs_h"][0][j])
                abs_subseg_post_auc_s0[j-1].append(ivim_params_post["aucs_s0"][0][j])
                abs_subseg_post_auc_s0_l[j-1].append(ivim_params_post["aucs_s0_l"][0][j])
                abs_subseg_post_auc_s0_m[j-1].append(ivim_params_post["aucs_s0_m"][0][j])
                abs_subseg_post_auc_s0_h[j-1].append(ivim_params_post["aucs_s0_h"][0][j])
                #abs_subseg_post_curvature[j-1].append((ivim_params_post["curvatures"][0][j] -ivim_params_pre["curvatures"][0][j])/ ivim_params_pre["curvatures"][0][j])
                # abs_subseg_post_c20[j-1].append((ivim_params_post["c20s"][0][j] - ivim_params_pre["c20s"][0][j]) / ivim_params_pre["c20s"][0][j])
                # abs_subseg_post_c50[j-1].append((ivim_params_post["c50s"][0][j] - ivim_params_pre["c50s"][0][j])/ ivim_params_pre["c50s"][0][j])
                # abs_subseg_post_c100[j-1].append((ivim_params_post["c100s"][0][j] - ivim_params_pre["c100s"][0][j])/ ivim_params_pre["c100s"][0][j])
                for i in range(3):
                    abs_subseg_post_biexp[i][j-1].append(ivim_params_post["biexps"][i][0][j])
                for i in range(5):
                    abs_subseg_post_triexp[i][j-1].append(ivim_params_post["triexp"][i][0][j])
            
    #now get the correlation of each subregions dose with param changes:
    corrs_biexp = [[],[],[]]
    ps_biexp = [[],[],[]]
    corrs_triexp = [[],[],[],[],[]]
    ps_triexp = [[],[],[],[],[]]
    corrs_adc = []
    ps_adc = []
    corrs_auc = []
    ps_auc = []
    corrs_auc_l = []
    ps_auc_l = []
    corrs_auc_m = []
    ps_auc_m = []
    corrs_auc_h = []
    ps_auc_h = []
    corrs_auc_s0 = []
    ps_auc_s0 = []
    corrs_auc_s0_l = []
    ps_auc_s0_l = []
    corrs_auc_s0_m = []
    ps_auc_s0_m = []
    corrs_auc_s0_h = []
    ps_auc_s0_h = []

    slope_dose_biexp = [[],[],[]]
    scores_biexp = [[],[],[]]
    slope_dose_triexp = [[],[],[],[],[]]
    scores_triexp = [[],[],[],[],[]]
    slope_dose_adc = []
    scores_adc = []
    slope_dose_auc = []
    scores_auc = []
    slope_dose_auc_l = []
    scores_auc_l = []
    slope_dose_auc_m = []
    scores_auc_m = []
    slope_dose_auc_h = []
    scores_auc_h = []
    slope_dose_auc_s0 = []
    scores_auc_s0 = []
    slope_dose_auc_s0_l = []
    scores_auc_s0_l = []
    slope_dose_auc_s0_m = []
    scores_auc_s0_m = []
    slope_dose_auc_s0_h = []
    scores_auc_s0_h = []
    ps_adc = []
    ps_auc_s0_l = []
    ps_auc_s0 = []

    for j in range(9):
        scores_biexp = []
        scores_triexp = []
        corr_biexp = []
        p_biexp = []
        corr_triexp = []
        p_triexp = []
        subseg_doses[j] = np.array(subseg_doses[j]).reshape(-1,1)

        corr_adc, p_adc = spearmanr(subseg_doses[j], subseg_changes_adc[j])
        corrs_adc.append(np.abs(corr_adc))
        X = np.column_stack((subseg_doses[j], abs_subseg_pre_adc[j]))
        y = (abs_subseg_post_adc[j]- adc_stats[0][j]) / adc_stats[1][j]
        model = LinearRegression()
        model.fit(X, y)
        slope_dose_adc.append(model.coef_[0])
        scores_adc.append(model.score(X,y))
        ps_adc.append(p_adc)

        corr_auc, p_auc = spearmanr(subseg_doses[j], subseg_changes_auc[j])
        corrs_auc.append(np.abs(corr_auc))
        X = np.column_stack((subseg_doses[j], abs_subseg_pre_auc[j]))
        y = (abs_subseg_post_auc[j]- auc_stats[0][j]) / auc_stats[1][j]
        model = LinearRegression()
        model.fit(X, y)
        slope_dose_auc.append(model.coef_[0])
        scores_auc.append(model.score(X,y))

        corr_auc_l, p_auc_l = spearmanr(subseg_doses[j], subseg_changes_auc_l[j])
        corrs_auc_l.append(np.abs(corr_auc_l))
        X = np.column_stack((subseg_doses[j], abs_subseg_pre_auc_l[j]))
        y = (abs_subseg_post_auc_l[j]- auc_l_stats[0][j]) / auc_l_stats[1][j]
        model = LinearRegression()
        model.fit(X, y)
        slope_dose_auc_l.append(model.coef_[0])
        scores_auc_l.append(model.score(X,y))

        corr_auc_m, p_auc_m = spearmanr(subseg_doses[j], subseg_changes_auc_m[j])
        corrs_auc_m.append(np.abs(corr_auc_m))
        X = np.column_stack((subseg_doses[j], abs_subseg_pre_auc_m[j]))
        y = (abs_subseg_post_auc_m[j]- auc_m_stats[0][j]) / auc_m_stats[1][j]
        model = LinearRegression()
        model.fit(X, y)
        slope_dose_auc_m.append(model.coef_[0])
        scores_auc_m.append(model.score(X,y))
        

        corr_auc_h, p_auc_h = spearmanr(subseg_doses[j], subseg_changes_auc_h[j])
        corrs_auc_h.append(np.abs(corr_auc_h))
        X = np.column_stack((subseg_doses[j], abs_subseg_pre_auc_h[j]))
        y = (abs_subseg_post_auc_h[j]- auc_h_stats[0][j]) / auc_h_stats[1][j]
        model = LinearRegression()
        model.fit(X, y)
        slope_dose_auc_h.append(model.coef_[0])
        scores_auc_h.append(model.score(X,y))

        corr_auc_s0, p_auc_s0 = spearmanr(subseg_doses[j], subseg_changes_auc_s0[j])
        corrs_auc_s0.append(np.abs(corr_auc_s0))
        X = np.column_stack((subseg_doses[j], abs_subseg_pre_auc_s0[j]))
        y = (abs_subseg_post_auc_s0[j]- auc_s0_stats[0][j]) / auc_s0_stats[1][j]
        model = LinearRegression()
        model.fit(X, y)
        slope_dose_auc_s0.append(model.coef_[0])
        scores_auc_s0.append(model.score(X,y))
        ps_auc_s0.append(p_auc_s0)

        corr_auc_s0_l, p_auc_s0_l = spearmanr(subseg_doses[j], subseg_changes_auc_s0_l[j])
        corrs_auc_s0_l.append(np.abs(corr_auc_s0_l))
        X = np.column_stack((subseg_doses[j], abs_subseg_pre_auc_s0_l[j]))
        y = (abs_subseg_post_auc_s0_l[j]- auc_s0_l_stats[0][j]) / auc_s0_m_stats[1][j]
        model = LinearRegression()
        model.fit(X, y)
        slope_dose_auc_s0_l.append(model.coef_[0])
        scores_auc_s0_l.append(model.score(X,y))
        ps_auc_s0_l.append(p_auc_s0_l)

        corr_auc_s0_m, p_auc_s0_m = spearmanr(subseg_doses[j], subseg_changes_auc_s0_m[j])
        corrs_auc_s0_m.append(np.abs(corr_auc_s0_m))
        X = np.column_stack((subseg_doses[j], abs_subseg_pre_auc_s0_m[j]))
        y = (abs_subseg_post_auc_s0_m[j]- auc_s0_m_stats[0][j]) / auc_s0_m_stats[1][j]
        model = LinearRegression()
        model.fit(X, y)
        slope_dose_auc_s0_m.append(model.coef_[0])
        scores_auc_s0_m.append(model.score(X,y))

        corr_auc_s0_h, p_auc_s0_h = spearmanr(subseg_doses[j], subseg_changes_auc_s0_h[j])
        corrs_auc_s0_h.append(np.abs(corr_auc_s0_h))
        X = np.column_stack((subseg_doses[j], abs_subseg_pre_auc_s0_h[j]))
        y = (abs_subseg_post_auc_s0_h[j]- auc_s0_h_stats[0][j]) / auc_s0_h_stats[1][j]
        model = LinearRegression()
        model.fit(X, y)
        slope_dose_auc_s0_h.append(model.coef_[0])
        scores_auc_s0_h.append(model.score(X,y))

        # print(f"corr ADC: {np.mean(corrs_adc)}")
        # print(f"corr biexp: {np.mean(corrs_adc)}")

        for i in range(3):
            X = np.column_stack((subseg_doses[j], abs_subseg_pre_biexp[i][j]))
            y = (abs_subseg_post_biexp[i][j]- biexp_stats[i][0][j]) / biexp_stats[i][1][j]
            model = LinearRegression()
            model.fit(X, y)
            score_biexp.append(model.score(X,y))
            slope_dose_biexp.append(model.coef_[0])

            corr, p = spearmanr(subseg_doses[j], subseg_changes_biexp[i][j])
            corrs_biexp[i].append(np.abs(corr))
            ps_biexp[i].append(p)

        for i in range(5):
            X = np.column_stack((subseg_doses[j], abs_subseg_pre_triexp[i][j]))
            y = (abs_subseg_post_triexp[i][j]- triexp_stats[i][0][j]) / triexp_stats[i][1][j]
            model = LinearRegression()
            model.fit(X, y)
            score_triexp.append(model.score(X,y))
            slope_dose_triexp.append(model.coef_[0])

            corr, p = spearmanr(subseg_doses[j], subseg_changes_triexp[i][j])
            corrs_triexp[i].append(np.abs(corr))
            ps_triexp[i].append(p)

    #now can make a bar plot for each, to show which subsegment has the strongest correlation w/ dose
    fig, ax = plt.subplots(figsize=(8,4))
    all_scores = corrs_adc
    cmap = plt.get_cmap('jet')
    normalize = plt.Normalize(vmin=np.amin(all_scores), vmax=np.amax(all_scores))
    colors = cmap(normalize(all_scores))


    bars = ax.bar(range(1,10), all_scores, color="coral", edgecolor="orangered")
    for bar, p_value in zip(bars, ps_adc):
        ax.text(bar.get_x() + bar.get_width() / 2, 1.05*bar.get_height(), f'p = {round(p_value,3)}', ha='center', va='bottom', rotation='vertical')

    ax.set_ylabel("$r_s$")
    ax.set_xticks(range(1, 10))
    ax.set_yticks(np.linspace(0, 1, 11))

    ax.set_xticklabels(["$A_1$", "$A_2$", "$A_3$", "$C_1$", "$C_2$", "$C_3$", "$S_1$", "$S_2$","$S_3$"], rotation=45, ha='right')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()

    #now do with auc_s0
    fig, ax = plt.subplots(figsize=(8,4))
    all_scores = corrs_auc_s0
    cmap = plt.get_cmap('jet')
    normalize = plt.Normalize(vmin=np.amin(all_scores), vmax=np.amax(all_scores))
    colors = cmap(normalize(all_scores))


    bars = ax.bar(range(1,10), all_scores, color="orchid", edgecolor="mediumvioletred")
    for bar, p_value in zip(bars, ps_auc_s0):
        ax.text(bar.get_x() + bar.get_width() / 2, 1.05*bar.get_height(), f'p = {round(p_value,3)}', ha='center', va='bottom', rotation='vertical')

    ax.set_ylabel("$r_s$")
    ax.set_xticks(range(1, 10))
    ax.set_yticks(np.linspace(0, 1, 11))

    ax.set_xticklabels(["$A_1$", "$A_2$", "$A_3$", "$C_1$", "$C_2$", "$C_3$", "$S_1$", "$S_2$","$S_3$"], rotation=45, ha='right')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()


    #now do with auc_s0
    fig, ax = plt.subplots(figsize=(8,4))
    all_scores = corrs_auc_s0_l
    cmap = plt.get_cmap('jet')
    normalize = plt.Normalize(vmin=np.amin(all_scores), vmax=np.amax(all_scores))
    colors = cmap(normalize(all_scores))


    bars = ax.bar(range(1,10), all_scores, color="indianred", edgecolor="firebrick")
    for bar, p_value in zip(bars, ps_auc_s0_l):
        ax.text(bar.get_x() + bar.get_width() / 2, 1.05*bar.get_height(), f'p = {round(p_value,3)}', ha='center', va='bottom', rotation='vertical')

    ax.set_ylabel("$r_s$")
    ax.set_xticks(range(1, 10))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xticklabels(["$A_1$", "$A_2$", "$A_3$", "$C_1$", "$C_2$", "$C_3$", "$S_1$", "$S_2$","$S_3$"], rotation=45, ha='right')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()

    #now do with slope for dose term
    fig, ax = plt.subplots(figsize=(8,4))
    all_scores = slope_dose_adc
    cmap = plt.get_cmap('jet')
    normalize = plt.Normalize(vmin=np.amin(all_scores), vmax=np.amax(all_scores))
    colors = cmap(normalize(all_scores))
    bars = ax.bar(range(1,10), all_scores, color="salmon", edgecolor="saddlebrown")
    ax.set_ylabel(r"$\frac{\Delta \, Z_f}{\Delta \, D}$", fontsize=20)
    ax.set_xticks(range(1, 10))
    #ax.set_yticks(np.linspace(0, 1, 11))

    ax.set_xticklabels(["$A_1$", "$A_2$", "$A_3$", "$C_1$", "$C_2$", "$C_3$", "$S_1$", "$S_2$","$S_3$"], rotation=45, ha='right')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()

    #now do with slope for dose term
    fig, ax = plt.subplots(figsize=(8,4))
    all_scores = slope_dose_auc_s0
    cmap = plt.get_cmap('jet')
    normalize = plt.Normalize(vmin=np.amin(all_scores), vmax=np.amax(all_scores))
    colors = cmap(normalize(all_scores))
    bars = ax.bar(range(1,10), all_scores, color="palevioletred", edgecolor="darkmagenta")
    ax.set_ylabel(r"$\frac{\Delta \, Z_f}{\Delta \, D}$", fontsize=20)
    ax.set_xticks(range(1, 10))
    #ax.set_yticks(np.linspace(0, 1, 11))

    ax.set_xticklabels(["$A_1$", "$A_2$", "$A_3$", "$C_1$", "$C_2$", "$C_3$", "$S_1$", "$S_2$","$S_3$"], rotation=45, ha='right')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()

    
    #now do with slope for dose term
    fig, ax = plt.subplots(figsize=(8,4))
    all_scores = slope_dose_auc_s0_l
    cmap = plt.get_cmap('jet')
    normalize = plt.Normalize(vmin=np.amin(all_scores), vmax=np.amax(all_scores))
    colors = cmap(normalize(all_scores))
    bars = ax.bar(range(1,10), all_scores, color="lightcoral", edgecolor="brown")
    ax.set_ylabel(r"$\frac{\Delta \, Z_f}{\Delta \, D}$", fontsize=20)
    ax.set_xticks(range(1, 10))
    #ax.set_yticks(np.linspace(0, 1, 11))

    ax.set_xticklabels(["$A_1$", "$A_2$", "$A_3$", "$C_1$", "$C_2$", "$C_3$", "$S_1$", "$S_2$","$S_3$"], rotation=45, ha='right')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()



    #now do this with all all_subsegs in one collection. 
    all_subseg_changes_biexp = [[],[],[]]
    all_subseg_changes_triexp = [[],[],[],[],[]]
    all_subseg_changes_adc = []
    all_subseg_changes_auc = []
    all_subseg_changes_auc_l = []
    all_subseg_changes_auc_m = []
    all_subseg_changes_auc_h = []
    all_subseg_changes_auc_s0 = []
    all_subseg_changes_auc_s0_l = []
    all_subseg_changes_auc_s0_m = []
    all_subseg_changes_auc_s0_h = []
    all_subseg_doses = []
    for patient in patient_nums:
        patient_dose_dict = "SIVIM" + str(patient)
        for roi in rois: 
            try:
                ivim_params_pre = ivim_stats_pre[patient][roi]
                ivim_params_post = ivim_stats_post[patient][roi]
                dose_stat = dose_stats[patient_dose_dict][roi]
            except:
                print(f"Failed for {roi} for {patient}")
                continue

            for ss in ["si", "ap", "ml"]:
                for j in range(3):
                    all_subseg_doses.append(dose_stat[ss][j][0])

            for j in range(1,10):
                
                
                all_subseg_changes_adc.append((ivim_params_post["adcs"][0][j] -ivim_params_pre["adcs"][0][j])/ ivim_params_pre["adcs"][0][j])
                all_subseg_changes_auc.append((ivim_params_post["aucs"][0][j] -ivim_params_pre["aucs"][0][j])/ ivim_params_pre["aucs"][0][j])
                all_subseg_changes_auc_l.append((ivim_params_post["aucs_l"][0][j] - ivim_params_pre["aucs_l"][0][j])/ ivim_params_pre["aucs_l"][0][j])
                all_subseg_changes_auc_m.append((ivim_params_post["aucs_m"][0][j] - ivim_params_pre["aucs_m"][0][j])/ ivim_params_pre["aucs_m"][0][j])
                all_subseg_changes_auc_h.append((ivim_params_post["aucs_h"][0][j] - ivim_params_pre["aucs_h"][0][j])/ ivim_params_pre["aucs_h"][0][j])
                all_subseg_changes_auc_s0.append((ivim_params_post["aucs_s0"][0][j] -ivim_params_pre["aucs_s0"][0][j])/ ivim_params_pre["aucs_s0"][0][j])
                all_subseg_changes_auc_s0_l.append((ivim_params_post["aucs_s0_l"][0][j] - ivim_params_pre["aucs_s0_l"][0][j])/ ivim_params_pre["aucs_s0_l"][0][j])
                all_subseg_changes_auc_s0_m.append((ivim_params_post["aucs_s0_m"][0][j] - ivim_params_pre["aucs_s0_m"][0][j])/ ivim_params_pre["aucs_s0_m"][0][j])
                all_subseg_changes_auc_s0_h.append((ivim_params_post["aucs_s0_h"][0][j] - ivim_params_pre["aucs_s0_h"][0][j])/ ivim_params_pre["aucs_s0_h"][0][j])
                for i in range(3):
                    all_subseg_changes_biexp[i].append((ivim_params_post["biexps"][i][0][j] - ivim_params_pre["biexps"][i][0][j])/ ivim_params_pre["biexps"][i][0][j])
                for i in range(5):
                    all_subseg_changes_triexp[i].append((ivim_params_post["triexp"][i][0][j] - ivim_params_pre["triexp"][i][0][j])/ ivim_params_pre["triexp"][i][0][j])
            

    scores_biexp = []
    scores_triexp = []
    corr_biexp = []
    p_biexp = []
    corr_triexp = []
    p_triexp = []
    
    all_subseg_doses = np.array(all_subseg_doses).reshape(-1,1)

    corr_adc, p_adc = spearmanr(all_subseg_doses, all_subseg_changes_adc)
    corr_adc = np.abs(corr_adc)

    corr_auc, p_auc = spearmanr(all_subseg_doses, all_subseg_changes_auc)
    corr_auc = np.abs(corr_auc)

    corr_auc_l, p_auc_l = spearmanr(all_subseg_doses, all_subseg_changes_auc_l)
    corr_auc_l = np.abs(corr_auc_l)

    corr_auc_m, p_auc_m = spearmanr(all_subseg_doses, all_subseg_changes_auc_m)
    corr_auc_m = np.abs(corr_auc_m)

    corr_auc_h, p_auc_h = spearmanr(all_subseg_doses, all_subseg_changes_auc_h)
    corr_auc_h = np.abs(corr_auc_h)

    corr_auc_s0, p_auc_s0 = spearmanr(all_subseg_doses, all_subseg_changes_auc_s0)
    corr_auc_s0 = np.abs(corr_auc_s0)

    corr_auc_s0_l, p_auc_s0_l = spearmanr(all_subseg_doses, all_subseg_changes_auc_s0_l)
    corr_auc_s0_l = np.abs(corr_auc_s0_l)

    corr_auc_s0_m, p_auc_s0_m = spearmanr(all_subseg_doses, all_subseg_changes_auc_s0_m)
    corr_auc_s0_m = np.abs(corr_auc_s0_m)

    corr_auc_s0_h, p_auc_s0_h = spearmanr(all_subseg_doses, all_subseg_changes_auc_s0_h)
    corr_auc_s0_h = np.abs(corr_auc_s0_h)

    print(f"Corr ADC: {corr_adc} | p ADC {p_adc}")
    print(f"Corr AUC: {corr_auc} | p AUC {p_auc}")
    print(f"Corr AUCl: {corr_auc_l} | p AUCl {p_auc_l}")
    print(f"Corr AUCm: {corr_auc_m} | p AUCm {p_auc_m}")
    print(f"Corr AUCh: {corr_auc_h} | p AUCh {p_auc_h}")
    
    print(f"Corr auc_s0: {corr_auc_s0} | p auc_s0 {p_auc_s0}")
    print(f"Corr auc_s0l: {corr_auc_s0_l} | p auc_s0l {p_auc_s0_l}")
    print(f"Corr auc_s0m: {corr_auc_s0_m} | p auc_s0m {p_auc_s0_m}")
    print(f"Corr auc_s0h: {corr_auc_s0_h} | p auc_s0h {p_auc_s0_h}")

    for i in range(3):
        corr, p = spearmanr(all_subseg_doses, all_subseg_changes_biexp[i])
        corr_biexp.append(np.abs(corr))
        p_biexp.append(p)
        print(f"Corr biexp: {corr} | p auc_s0 {p}")

    for i in range(5):
        corr, p = spearmanr(all_subseg_doses, all_subseg_changes_triexp[i])
        corr_triexp.append(np.abs(corr))
        p_triexp.append(p)
        print(f"Corr triexp: {corr} | p triexp {p}")
    import colorbrewer as cb

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    labels=["$A_1$", "$A_2$", "$A_3$", "$C_1$", "$C_2$", "$C_3$", "$S_1$", "$S_2$","$S_3$"]
    fig, ax = plt.subplots(figsize=(8,8))
    for j in range(len(all_subseg_doses)):
        if j < 9:
            ax.scatter(all_subseg_doses[j], all_subseg_changes_adc[j], c=colors[j % 9], label=labels[j % 9])
        else: 
            ax.scatter(all_subseg_doses[j], all_subseg_changes_adc[j], c=colors[j % 9])
    ax.set_ylim([0,3.5])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Dose (Gy)", fontsize=12)
    ax.set_ylabel(r"$\frac{\Delta \, ADC}{ADC_{initial}}$", fontsize=16)

    ax.legend(loc="upper left")
    plt.show()


    fig, ax = plt.subplots(figsize=(8,8))
    for j in range(len(all_subseg_doses)):
        if j < 9:
            ax.scatter(all_subseg_doses[j], all_subseg_changes_auc_s0[j], c=colors[j % 9], label=labels[j % 9])
        else: 
            ax.scatter(all_subseg_doses[j], all_subseg_changes_auc_s0[j], c=colors[j % 9])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Dose (Gy)", fontsize=12)
    ax.set_ylabel(r"$\frac{\Delta \, AUC}{AUC_{initial}}$", fontsize=16)
    ax.set_ylim([-0.3,0.05])
    ax.legend(loc="lower left")


    plt.show()


    fig, ax = plt.subplots(figsize=(8,8))
    for j in range(len(all_subseg_doses)):
        if j < 9:
            ax.scatter(all_subseg_doses[j], all_subseg_changes_auc_s0_l[j], c=colors[j % 9], label=labels[j % 9])
        else: 
            ax.scatter(all_subseg_doses[j], all_subseg_changes_auc_s0_l[j], c=colors[j % 9])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Dose (Gy)", fontsize=12)
    ax.set_ylabel(r"$\frac{\Delta \, AUC_{L}}{AUC_{L, initial}}$", fontsize=16)
    ax.set_ylim([-0.06,0.08])

    ax.legend(loc="lower left")
    plt.show()

    # fig, ax = plt.subplots(figsize=(20,20))
    # ax.scatter(whole_doses, whole_changes_auc_s0_l, c="palegreen")
    
    # plt.show()



def get_parameter_svd_variance(deblurred=True, smoothed=False):
    #first need to run get_all_parotid_param_and_dose_stats. dose stats were calculated previously in cross modality code.
    
    all_hstacks = [] #store each roi flattened hstack to be vstacked for the whole set
    for scan_type in ["pre", "post"]:
        if scan_type == "pre":
            patient_nums = ["10", "11", "12", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']
        else:
            patient_nums = ["10", "11", "12", "13", "16"]

        for patient in patient_nums: 
            img_series_path = os.path.join(os.getcwd(), "data_mri")
            img_series_path = os.path.join(img_series_path, str("SIVIM" + patient + scan_type + "diffusion_imgs"))
            with open(img_series_path, "rb") as fp:
                img_dict_full = pickle.load(fp)
            img_dict = img_dict_full["diffusion"]

            for roi in ["r_par", "l_par"]:
                # auc_img = getattr(img_dict, f"auc_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                # adc_img = getattr(img_dict, f"adc_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                # curve_img = getattr(img_dict, f"curve_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                # triexp_img = getattr(img_dict, f"triexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                # biexp_img = getattr(img_dict, f"biexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                # curve20 = getattr(img_dict, f"c20_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                # curve50 = getattr(img_dict, f"c50_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                # curve100 = getattr(img_dict, f"c100_img_{roi}_{deblurred}_{smoothed}_{scan_type}")

                # img_stack = np.hstack((biexp_img[1,...].reshape(-1), biexp_img[0,...].reshape(-1), biexp_img[2,...].reshape(-1), triexp_img[0,...].reshape(-1), triexp_img[1,...].reshape(-1), triexp_img[2,...].reshape(-1), triexp_img[3,...].reshape(-1), triexp_img[4,...].reshape(-1), adc_img.reshape(-1), auc_img.reshape(-1), curve_img.reshape(-1), curve20.reshape(-1), curve50.reshape(-1), curve100.reshape(-1) ))
                # all_hstacks.append(img_stack)
                try:
                    img = getattr(img_dict, f"deconv_array_{roi}")[0,...]
                    auc_img = getattr(img_dict, f"auc_img_{roi}_{deblurred}_{smoothed}_{False}_{scan_type}")
                    auc_l_img = getattr(img_dict, f"auc_l_img_{roi}_{deblurred}_{smoothed}_{False}_{scan_type}")
                    auc_m_img = getattr(img_dict, f"auc_m_img_{roi}_{deblurred}_{smoothed}_{False}_{scan_type}")
                    auc_h_img = getattr(img_dict, f"auc_h_img_{roi}_{deblurred}_{smoothed}_{False}_{scan_type}")

                    auc_s0_img = getattr(img_dict, f"auc_img_{roi}_{deblurred}_{smoothed}_{True}_{scan_type}")
                    auc_s0_l_img = getattr(img_dict, f"auc_l_img_{roi}_{deblurred}_{smoothed}_{True}_{scan_type}")
                    auc_s0_m_img = getattr(img_dict, f"auc_m_img_{roi}_{deblurred}_{smoothed}_{True}_{scan_type}")
                    auc_s0_h_img = getattr(img_dict, f"auc_h_img_{roi}_{deblurred}_{smoothed}_{True}_{scan_type}")
                    adc_img = getattr(img_dict, f"adc_img_{roi}_{deblurred}_{smoothed}_{scan_type}")[0,:,:,:]
                    triexp_img = getattr(img_dict, f"triexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    biexp_img = getattr(img_dict, f"biexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
            
                    img_stack = np.hstack((biexp_img[1,...].reshape(-1,1), biexp_img[0,...].reshape(-1,1), biexp_img[2,...].reshape(-1,1), triexp_img[0,...].reshape(-1,1), triexp_img[1,...].reshape(-1,1), triexp_img[2,...].reshape(-1,1), triexp_img[3,...].reshape(-1,1), triexp_img[4,...].reshape(-1,1), adc_img.reshape(-1,1),  auc_s0_img.reshape(-1,1), auc_s0_l_img.reshape(-1,1), auc_s0_m_img.reshape(-1,1), auc_s0_h_img.reshape(-1,1))) 
                    
                    outside_rows = (img.reshape(-1,1) < 40).any(axis=1)
                    img_stack = img_stack[~outside_rows]

                    nan_rows = np.isnan(img_stack).any(axis=1)
                    img_stack = img_stack[~nan_rows]
                    
                    neg_rows = np.any(img_stack < 0, axis=1)
                    img_stack = img_stack[~neg_rows]

                    large_rows = np.any(img_stack > 1000, axis=1)
                    img_stack = img_stack[~large_rows]

        

                    all_hstacks.append(img_stack)
                except:
                    continue
    #now want to get the predictive power of the whole mean dose for whole mean IVIM change first. 

    X = np.vstack(all_hstacks)

    #X = X[:, ::-1]
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X = (X - mu) / sigma



    corrs = np.corrcoef(X, rowvar=False)
    #corrs = np.tril(corrs)
    corrs = np.abs(corrs)
    #corrs[np.eye(corrs.shape[0], dtype=bool)] = np.nan
    corrs[corrs==0] = np.nan
    fig, ax = plt.subplots(figsize=(15,15))
    im = ax.imshow(corrs, cmap="viridis")
    params = ["$D_{biexp}$", "$D*_{biexp}$", "$f_{biexp}$", "$D_{triexp}$",  "$D*_1$","$f_1$", "$D*_2$",  "$f_2$", "ADC", "AUC", "$AUC_L$", "$AUC_M$", "$AUC_H$"]
    ax.set_xticks(range(13))
    ax.set_yticks(range(13))
    ax.set_xticklabels(params, fontsize=20, rotation=60)
    ax.set_yticklabels(params, fontsize=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.colorbar(im, ax=ax)
    plt.show()



    U, S, VT = np.linalg.svd(X, full_matrices=0)
    all_scores = np.zeros((13))      
    for j in range(1):   #top imp features
        pc = VT[j,:]**2#*S[j] #squared to get variance captured
        all_scores += np.abs(pc)
    all_scores /= np.sum(all_scores)
    params = ["$D_{biexp}$", "$D*_{biexp}$", "$f_{biexp}$", "$D_{triexp}$",  "$D*_1$", "$f_1$", "$D*_2$",  "$f_2$", "ADC", "AUC", "$AUC_L$", "$AUC_M$", "$AUC_H$"]
    fig, ax = plt.subplots(figsize=(15,15))
    cmap = plt.get_cmap('jet')
    normalize = plt.Normalize(vmin=min(all_scores), vmax=max(all_scores))
    colors = cmap(normalize(all_scores))
    ax.bar(params, all_scores, color=colors)
    ax.set_ylabel("Relative Variance Captured", fontsize=20)
    ax.set_xticklabels(params, rotation=45, ha='right')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation='vertical')
    plt.show()

def plot_ivim_hists_blur_and_deblur(scan_type="pre"):    
    save_path = os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", str("param_roi_hists_" + scan_type + "_" + "True" + ".txt"))
    with open(save_path, "rb") as fp:
        all_stats_deblur = pickle.load(fp)
    save_path = os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", str("param_roi_hists_" + scan_type + "_" + "False" + ".txt"))
    with open(save_path, "rb") as fp:
        all_stats = pickle.load(fp)

    biexp_roi_hists_deblur = [[],[],[]] #for cord, will be whole gland values only
    triexp_roi_hists_deblur = [[],[],[],[],[]]
    auc_roi_hists_deblur = []
    curvature_roi_hists_deblur = []
    c20_roi_hists_deblur = []
    c50_roi_hists_deblur = []
    c100_roi_hists_deblur = []
    adc_roi_hists_deblur = []

    biexp_roi_hists = [[],[],[]] #for cord, will be whole gland values only
    triexp_roi_hists = [[],[],[],[],[]]
    auc_roi_hists = []
    curvature_roi_hists = []
    c20_roi_hists = []
    c50_roi_hists = []
    c100_roi_hists = []
    adc_roi_hists = []

    biexp_roi_bins = [[],[],[]] #for cord, will be whole gland values only
    triexp_roi_bins = [[],[],[],[],[]]
    auc_roi_bins = []
    curvature_roi_bins = []
    c20_roi_bins = []
    c50_roi_bins = []
    c100_roi_bins = []
    adc_roi_bins = []

    
    
    for patient in all_stats_deblur:
        for roi in ["r_par", "l_par"]:
            try:
                for i in range(3):
                    biexp_roi_hists_deblur[i].append(all_stats_deblur[patient][roi]["biexps"][i][0][0])
                    biexp_roi_hists[i].append(all_stats[patient][roi]["biexps"][i][0][0])
                    biexp_roi_bins[i].append(all_stats_deblur[patient][roi]["biexps"][i][0][1])
                for i in range(5):
                    triexp_roi_hists_deblur[i].append(all_stats_deblur[patient][roi]["triexp"][i][0][0])
                    triexp_roi_hists[i].append(all_stats[patient][roi]["triexp"][i][0][0])
                    triexp_roi_bins[i].append(all_stats_deblur[patient][roi]["triexp"][i][0][1])

                auc_roi_hists_deblur.append(all_stats_deblur[patient][roi]["aucs"][0][0])
                adc_roi_hists_deblur.append(all_stats_deblur[patient][roi]["adcs"][0][0])
                curvature_roi_hists_deblur.append(all_stats_deblur[patient][roi]["curvatures"][0][0])
                c20_roi_hists_deblur.append(all_stats_deblur[patient][roi]["c20s"][0][0])
                c50_roi_hists_deblur.append(all_stats_deblur[patient][roi]["c50s"][0][0])
                c100_roi_hists_deblur.append(all_stats_deblur[patient][roi]["c100s"][0][0])

                auc_roi_hists.append(all_stats[patient][roi]["aucs"][0][0])
                adc_roi_hists.append(all_stats[patient][roi]["adcs"][0][0])
                curvature_roi_hists.append(all_stats[patient][roi]["curvatures"][0][0])
                c20_roi_hists.append(all_stats[patient][roi]["c20s"][0][0])
                c50_roi_hists.append(all_stats[patient][roi]["c50s"][0][0])
                c100_roi_hists.append(all_stats[patient][roi]["c100s"][0][0])

                auc_roi_bins.append(all_stats[patient][roi]["aucs"][0][1])
                adc_roi_bins.append(all_stats[patient][roi]["adcs"][0][1])
                curvature_roi_bins.append(all_stats[patient][roi]["curvatures"][0][1])
                c20_roi_bins.append(all_stats[patient][roi]["c20s"][0][1])
                c50_roi_bins.append(all_stats[patient][roi]["c50s"][0][1])
                c100_roi_bins.append(all_stats[patient][roi]["c100s"][0][1])
            except:
                continue

    #now for each, get avg. 
    auc_roi_hist = np.mean(np.array(auc_roi_hists), axis=0)
    adc_roi_hist = np.mean(np.array(adc_roi_hists), axis=0)
    curvature_roi_hist = np.mean(np.array(curvature_roi_hists), axis=0)
    c20_roi_hist = np.mean(np.array(c20_roi_hists), axis=0)
    c50_roi_hist = np.mean(np.array(c50_roi_hists), axis=0)
    c100_roi_hist = np.mean(np.array(c100_roi_hists), axis=0)
    biexp_roi_hist = []
    triexp_roi_hist = []
    for i in range(3):
        biexp_roi_hist.append(np.mean(np.array(biexp_roi_hists[i]), axis=0))
    for i in range(5):
        triexp_roi_hist.append(np.mean(np.array(triexp_roi_hists[i]), axis=0))

    auc_roi_hist_deblur = np.mean(np.array(auc_roi_hists_deblur), axis=0)
    adc_roi_hist_deblur = np.mean(np.array(adc_roi_hists_deblur), axis=0)
    curvature_roi_hist_deblur = np.mean(np.array(curvature_roi_hists_deblur), axis=0)
    c20_roi_hist_deblur = np.mean(np.array(c20_roi_hists_deblur), axis=0)
    c50_roi_hist_deblur = np.mean(np.array(c50_roi_hists_deblur), axis=0)
    c100_roi_hist_deblur = np.mean(np.array(c100_roi_hists_deblur), axis=0)
    biexp_roi_hist_deblur = []
    triexp_roi_hist_deblur = []
    for i in range(3):
        biexp_roi_hist_deblur.append(np.mean(np.array(biexp_roi_hists_deblur[i]), axis=0))
    for i in range(5):
        triexp_roi_hist_deblur.append(np.mean(np.array(triexp_roi_hists_deblur[i]), axis=0))

    #first make biexp plots
    coefs = ["$D*$", "D", "f"]
    for i in range(3):
        hist = biexp_roi_hist[i]
        hist_deblur = biexp_roi_hist_deblur[i]
        hist /= np.sum(hist)
        hist_deblur /= np.sum(hist_deblur)
        bins = biexp_roi_bins[i][0]
        bin_centres = (bins[1:] + bins[:-1])*0.5
        bin_width = bins[1] - bins[0]
        x1 = bin_centres - bin_width / 4
        x2 = bin_centres + bin_width / 4

        fig, ax = plt.subplots()
        ax.bar(x1, hist_deblur, width=bin_width/2, color="mediumpurple", alpha=0.7)
        ax.bar(x2, hist, width=bin_width/2, color="orangered", alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(coefs[i])
        ax.set_ylabel("Relative Proportion")
        plt.show()

    coefs = ["D", "$D*_1$", "$D*_2$", "$f_1$", "$f_2$"]
    #make triexp
    for i in range(5):
        hist = triexp_roi_hist[i]
        hist_deblur = triexp_roi_hist_deblur[i]
        hist /= np.sum(hist)
        hist_deblur /= np.sum(hist_deblur)
        bins = triexp_roi_bins[i][0]
        bin_centres = (bins[1:] + bins[:-1])*0.5
        bin_width = bins[1] - bins[0]
        x1 = bin_centres - bin_width / 4
        x2 = bin_centres + bin_width / 4

        fig, ax = plt.subplots()
        ax.bar(x1, hist_deblur, width=bin_width/2, color="mediumpurple", alpha=0.7)
        ax.bar(x2, hist, width=bin_width/2, color="orangered", alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(coefs[i])
        ax.set_ylabel("Relative Proportion")
        plt.show()
    

    #now adc 
    hist = adc_roi_hist
    hist_deblur = adc_roi_hist_deblur
    hist /= np.sum(hist)
    hist_deblur /= np.sum(hist_deblur)
    bins = adc_roi_bins[0]
    bin_centres = (bins[1:] + bins[:-1])*0.5
    bin_width = bins[1] - bins[0]
    x1 = bin_centres - bin_width / 4
    x2 = bin_centres + bin_width / 4

    fig, ax = plt.subplots()
    ax.bar(x1, hist_deblur, width=bin_width/2, color="mediumpurple", alpha=0.7)
    ax.bar(x2, hist, width=bin_width/2, color="orangered", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("ADC")
    ax.set_ylabel("Relative Proportion")
    plt.show()

    #now auc 
    hist = auc_roi_hist
    hist_deblur = auc_roi_hist_deblur
    hist /= np.sum(hist)
    hist_deblur /= np.sum(hist_deblur)
    bins = auc_roi_bins[0]
    bin_centres = (bins[1:] + bins[:-1])*0.5
    bin_width = bins[1] - bins[0]
    x1 = bin_centres - bin_width / 4
    x2 = bin_centres + bin_width / 4

    fig, ax = plt.subplots()
    ax.bar(x1, hist_deblur, width=bin_width/2, color="mediumpurple", alpha=0.7)
    ax.bar(x2, hist, width=bin_width/2, color="orangered", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("AUC")
    ax.set_ylabel("Relative Proportion")
    plt.show()

    #now curvature
    hist = curvature_roi_hist
    hist_deblur = curvature_roi_hist_deblur
    hist /= np.sum(hist)
    hist_deblur /= np.sum(hist_deblur)
    bins = curvature_roi_bins[0]
    bin_centres = (bins[1:] + bins[:-1])*0.5
    bin_width = bins[1] - bins[0]
    x1 = bin_centres - bin_width / 4
    x2 = bin_centres + bin_width / 4

    fig, ax = plt.subplots()
    ax.bar(x1, hist_deblur, width=bin_width/2, color="mediumpurple", alpha=0.7)
    ax.bar(x2, hist, width=bin_width/2, color="orangered", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("$b_{max curvature}$")
    ax.set_ylabel("Relative Proportion")
    plt.show()

    #now c20
    hist = c20_roi_hist
    hist_deblur = c20_roi_hist_deblur
    hist /= np.sum(hist)
    hist_deblur /= np.sum(hist_deblur)
    bins = c20_roi_bins[0]
    bin_centres = (bins[1:] + bins[:-1])*0.5
    bin_width = bins[1] - bins[0]
    x1 = bin_centres - bin_width / 4
    x2 = bin_centres + bin_width / 4

    fig, ax = plt.subplots()
    ax.bar(x1, hist_deblur, width=bin_width/2, color="mediumpurple", alpha=0.7)
    ax.bar(x2, hist, width=bin_width/2, color="orangered", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("$C_{20}$")
    ax.set_ylabel("Relative Proportion")
    plt.show()

    #now c50
    hist = c50_roi_hist
    hist_deblur = c50_roi_hist_deblur
    hist /= np.sum(hist)
    hist_deblur /= np.sum(hist_deblur)
    bins = c50_roi_bins[0]
    bin_centres = (bins[1:] + bins[:-1])*0.5
    bin_width = bins[1] - bins[0]
    x1 = bin_centres - bin_width / 4
    x2 = bin_centres + bin_width / 4

    fig, ax = plt.subplots()
    ax.bar(x1, hist_deblur, width=bin_width/2, color="mediumpurple", alpha=0.7)
    ax.bar(x2, hist, width=bin_width/2, color="orangered", alpha=0.7)

    ax.set_xlabel("$C_{50}$")
    ax.set_ylabel("Relative Proportion")
    plt.show()

    #now c100
    hist = c100_roi_hist
    hist_deblur = c100_roi_hist_deblur
    hist /= np.sum(hist)
    hist_deblur /= np.sum(hist_deblur)
    bins = c100_roi_bins[0]
    bin_centres = (bins[1:] + bins[:-1])*0.5
    bin_width = bins[1] - bins[0]
    x1 = bin_centres - bin_width / 4
    x2 = bin_centres + bin_width / 4

    fig, ax = plt.subplots()
    ax.bar(x1, hist_deblur, width=bin_width/2, color="mediumpurple", alpha=0.7)
    ax.bar(x2, hist, width=bin_width/2, color="orangered", alpha=0.7)

    ax.set_xlabel("$C_{100}$")
    ax.set_ylabel("Relative Proportion")
    plt.show()

def get_all_roi_voxels(deblurred=True, smoothed=False, scan_type="pre"):
     
    #dose stats already obtained 
    path = "/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep"
    #need to get all the different params for each different img type

    all_stats = {}

    #triexp order D, Dp1, Dp2, f1, f2
    #biexp order pD, D, f

    if scan_type == "pre":
        patient_nums = ["10", "11","12", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']
    else:
        patient_nums = ["10","11","12","13"]

    for patient_num in patient_nums:
        
        all_stats[patient_num] = {}



        img_series_path = os.path.join(os.getcwd(), "data_mri")
        img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + scan_type + "diffusion_imgs"))
        with open(img_series_path, "rb") as fp:
            img_dict_full = pickle.load(fp)
        img_dict = img_dict_full["diffusion"]

        

        biexp_cord_voxels = [[],[],[]] #for cord, will be whole gland values only
        triexp_cord_voxels = [[],[],[],[],[]]
        auc_cord_voxels = []
        curvature_cord_voxels = []
        c20_cord_voxels = []
        c50_cord_voxels = []
        c100_cord_voxels = []
        adc_cord_voxels = []

        with open(os.path.join(os.path.join(os.getcwd(), "data_mri"), str("sivim" + patient_num + "_" + scan_type + "_MR_mask_dict")), "rb") as fp:
            mask_dict = pickle.load(fp)
        
       

        roi_masks = {}
        for roi in ["cord", "l_par", "r_par"]:
            if roi not in mask_dict.keys():
                continue
            all_stats[patient_num][roi] = {}
            #load different parameter images
            
            
            if "cord" not in roi.lower():
                all_stats[patient_num][roi] = {}
                #subsegment as needed
                Chopper.organ_chopper(mask_dict[roi], [0,0,2], name="segmented_contours_si")
                Chopper.organ_chopper(mask_dict[roi], [0,2,0], name="segmented_contours_ap")
                Chopper.organ_chopper(mask_dict[roi], [2,0,0], name="segmented_contours_ml")
                if roi == "r_par":
                    mask_dict[roi].segmented_contours_ml.reverse()   #sort medial to lateral

                coords_array = getattr(img_dict, f"deconv_coords_{roi}")
                #plot_3d_image(mask_dict[roi].mask_deconv)
                get_contour_masks.get_deconv_structure_masks(coords_array, mask_dict[roi])
            else:

                try:
                    whole_mask = mask_dict[roi].mask_deconv.astype(bool)
                    whole_mask = mask_dict[roi].mask_deconv.astype(bool)
                    auc_img = getattr(img_dict, f"auc_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    adc_img = getattr(img_dict, f"adc_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    curve_img = getattr(img_dict, f"curve_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    c20_img = getattr(img_dict, f"c20_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    c50_img = getattr(img_dict, f"c50_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    c100_img = getattr(img_dict, f"c100_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    triexp_img = getattr(img_dict, f"triexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                    biexp_img = getattr(img_dict, f"biexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                except:
                    continue
               
                for i in range(3):
                    # plot_3d_image(biexp_img[i,...])
                    # plot_3d_image(whole_mask)
                    
                    img = biexp_img[i,...]
                    vals = img[whole_mask]
                    vals = vals[~np.isnan(vals)]

                    biexp_cord_voxels[i].append(vals)   #add all 3 biexp params



                all_stats[patient_num][roi]["biexps"] = biexp_cord_voxels

                
                #get triexp params:
                for i in range(5):

                    img = triexp_img[i,...]
                    vals = img[whole_mask]

                    vals = vals[~np.isnan(vals)]
                    
                    triexp_cord_voxels[i].append(vals)   #add all 5 triexp params

                all_stats[patient_num][roi]["triexp"] = triexp_cord_voxels


            #now auc image 
                img = auc_img
                vals = img[whole_mask]
                vals = vals[~np.isnan(vals)]
                auc_cord_voxels.append(vals)   
                all_stats[patient_num][roi]["aucs"] = auc_cord_voxels


            #now ADC image
                img = adc_img
                vals = img[whole_mask]
                vals = vals[~np.isnan(vals)]
                adc_cord_voxels.append(vals)  
                all_stats[patient_num][roi]["adcs"] = adc_cord_voxels


            #now curvature image
                img = curve_img
                vals = img[whole_mask]
                vals = vals[~np.isnan(vals)]
                curvature_cord_voxels.append(vals)   
                all_stats[patient_num][roi]["curvatures"] = curvature_cord_voxels


                img = c20_img
                vals = img[whole_mask]
                vals = vals[~np.isnan(vals)]
                c20_cord_voxels.append(vals)   
                all_stats[patient_num][roi]["c20s"] = c20_cord_voxels
                
                img = c50_img
                vals = img[whole_mask]
                vals = vals[~np.isnan(vals)]
                c50_cord_voxels.append(vals)   
                all_stats[patient_num][roi]["c50s"] = c50_cord_voxels

                img = c100_img
                vals = img[whole_mask]
                vals = vals[~np.isnan(vals)]
                c100_cord_voxels.append(vals)   
                all_stats[patient_num][roi]["c100s"] = c100_cord_voxels

                
                
                

        for roi in ["l_par", "r_par"]:
      

            try:
                whole_mask = mask_dict[roi].mask_deconv.astype(bool)
                auc_img = getattr(img_dict, f"auc_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                adc_img = getattr(img_dict, f"adc_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                curve_img = getattr(img_dict, f"curve_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                c20_img = getattr(img_dict, f"c20_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                c50_img = getattr(img_dict, f"c50_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                c100_img = getattr(img_dict, f"c100_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                triexp_img = getattr(img_dict, f"triexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
                biexp_img = getattr(img_dict, f"biexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
            except:
                continue

            biexp_roi_voxels = [[],[],[]] #for cord, will be whole gland values only
            triexp_roi_voxels = [[],[],[],[],[]]
            auc_roi_voxels = []
            curvature_roi_voxels = []
            c20_roi_voxels = []
            c50_roi_voxels = []
            c100_roi_voxels = []
            adc_roi_voxels = []

            #first add whole gland stats
            whole_mask = mask_dict[roi].mask_deconv.astype(bool)
            #get biexp params:
            for i in range(3):

                img = biexp_img[i,...]
                vals = img[whole_mask]
                vals = vals[~np.isnan(vals)]
                biexp_roi_voxels[i].append(vals)   #add all 3 biexp params

            #get triexp params:
            bin_bounds = [[2e-4,2e-3], [4e-3, 3e-2], [3e-2, 0.5], [0,0.4], [0,0.4]]
            for i in range(5):
                img = triexp_img[i,...]
                vals = img[whole_mask]
                vals = vals[~np.isnan(vals)]
                triexp_roi_voxels[i].append(vals) 

            #now auc image

            img = auc_img
            vals = img[whole_mask]
            vals = img[whole_mask]
            vals = vals[~np.isnan(vals)]
            auc_roi_voxels.append(vals)  

            #now ADC image
            img = adc_img
            vals = img[whole_mask]
            vals = img[whole_mask]
            vals = vals[~np.isnan(vals)]
            adc_roi_voxels.append(vals)  

            #now curvature image
            img = curve_img
            vals = img[whole_mask]
            vals = img[whole_mask]
            vals = vals[~np.isnan(vals)]
            curvature_roi_voxels.append(vals) 


            img = c20_img
            vals = img[whole_mask]
            vals = img[whole_mask]
            vals = vals[~np.isnan(vals)]
            c20_roi_voxels.append(vals)   


            img = c50_img
            vals = img[whole_mask]
            vals = img[whole_mask]
            vals = vals[~np.isnan(vals)]
            c50_roi_voxels.append(vals)   


            img = c100_img
            vals = img[whole_mask]
            vals = img[whole_mask]
            vals = vals[~np.isnan(vals)]
            c100_roi_voxels.append(vals)   


            for seg in ["si", "ap", "ml"]:

                masks = getattr(mask_dict[roi], f"subseg_masks_deconv_{seg}")
                for mask in masks:
                    mask = mask.astype(bool)
                    #get biexp params:

                    for i in range(3):

                        img = biexp_img[i,...]
                        vals = img[mask]
                        vals = vals[~np.isnan(vals)]
                        biexp_roi_voxels[i].append(vals)   #add all 3 biexp params

                    #get triexp params:
                    for i in range(5):
                        img = triexp_img[i,...]
                        vals = img[mask]
                        vals = vals[~np.isnan(vals)]
                        triexp_roi_voxels[i].append(vals)   #add all 3 biexp params

                    #now auc image
                    img = auc_img
                    vals = img[mask]

                    vals = vals[~np.isnan(vals)]
                    auc_roi_voxels.append(vals)  

                    #now ADC image
                    img = adc_img
                    vals = img[mask]
                    vals = vals[~np.isnan(vals)]
                    adc_roi_voxels.append(vals)  

                    #now curvature image
                    img = curve_img
                    vals = img[mask]
                    vals = vals[~np.isnan(vals)]
                    curvature_roi_voxels.append(vals) 

                    img = c20_img
                    vals = img[mask]
                    vals = vals[~np.isnan(vals)]
                    c20_roi_voxels.append(vals)   

                    img = c50_img
                    vals = img[mask]
                    vals = vals[~np.isnan(vals)]
                    c50_roi_voxels.append(vals)   

                    img = c100_img
                    vals = img[mask]
                    vals = vals[~np.isnan(vals)]
                    c100_roi_voxels.append(vals)   



            all_stats[patient_num][roi]["curvatures"] = curvature_roi_voxels
            all_stats[patient_num][roi]["c20s"] = c20_roi_voxels
            all_stats[patient_num][roi]["c50s"] = c50_roi_voxels
            all_stats[patient_num][roi]["c100s"] = c100_roi_voxels
            all_stats[patient_num][roi]["adcs"] = adc_roi_voxels
            all_stats[patient_num][roi]["aucs"] = auc_roi_voxels
            all_stats[patient_num][roi]["biexps"] = biexp_roi_voxels
            all_stats[patient_num][roi]["triexp"] = triexp_roi_voxels

    
    #

    save_path = os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", str("param_roi_voxels_" + scan_type + "_" + str(deblurred) + ".txt"))
    with open(save_path, "wb") as fp:
        pickle.dump(all_stats, fp)
   
def plot_avg_param_subsegs(deblurred=True, img_type="auc_s0"):
    save_path = os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", str("param_roi_stats_" + "pre" + "_" + str(deblurred) + ".txt"))
    with open(save_path, "rb") as fp:
        stats = pickle.load(fp)

    stats = stats["all"][img_type]
    stats_si = stats[0][1:4]
    stats_ap = stats[0][4:7]
    stats_ml = stats[0][7:]

    patient_num = "17"
    data_folder = os.path.join(os.getcwd(), "data_mri")

    with open(os.path.join(data_folder, str("sivim" + patient_num + "_pre_MR_mask_dict")), "rb") as fp:
        mask_dict = pickle.load(fp)["l_par"]
    Chopper.organ_chopper(mask_dict, [0,0,2], name="segmented_contours_si")
    Chopper.organ_chopper(mask_dict, [0,2,0], name="segmented_contours_ap")
    Chopper.organ_chopper(mask_dict, [2,0,0], name="segmented_contours_ml")
    ml_subsegs = mask_dict.segmented_contours_ml
    si_subsegs = mask_dict.segmented_contours_si
    ap_subsegs = mask_dict.segmented_contours_ap

    from Visuals_Subsegs import plotSubsegments
    plotSubsegments(si_subsegs, values=stats_si, min_val=641, max_val=680)
    plotSubsegments(ap_subsegs, values=stats_ap, min_val=641, max_val=680)
    plotSubsegments(ml_subsegs, values=stats_ml, min_val=641, max_val=680)

def get_cnrs(deblurred=True, smoothed=False, scan_type="pre"):
    #get cnrs of neighbouring par subsegs,. first run get_all_roi_voxels
    save_path = os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", str("param_roi_stats_" + scan_type + "_" + str(deblurred) + ".txt"))
    with open(save_path, "rb") as fp:
        all_stats = pickle.load(fp)
    if scan_type == "pre":
        patient_nums = ["10", "11","12", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']
    else:
        patient_nums = ["10","11","12","13", "16"]

    all_cnr_stats = {}
    all_cnr_avgs = {} #hold avg for each img type (across all neighbours)


    img_type = "biexps"
    all_cnr_stats["biexp"] = []
    all_cnr_avgs["biexp"] = []
    for i in range(3):
        all_cnr_stats["biexp"].append([])
        all_cnr_avgs["biexp"].append([])
        cnr_si1 = []
        cnr_si2 = []
        cnr_ap1 = []
        cnr_ap2 = []
        cnr_ml1 = []
        cnr_ml2 = []

        for patient in patient_nums:
            for roi in ["r_par", "l_par"]:
                try:
                    means = all_stats[patient][roi][img_type][i][0]
                    stds = all_stats[patient][roi][img_type][i][1]
                    cnr_si1.append((means[2] - means[1])/(0.5*(stds[2] + stds[1])))
                    cnr_si2.append((means[3] - means[2])/(0.5*(stds[3] + stds[2])))
                    cnr_ap1.append((means[5] - means[4])/(0.5*(stds[5] + stds[4])))
                    cnr_ap2.append((means[6] - means[5])/(0.5*(stds[6] + stds[5])))
                    cnr_ml1.append((means[8] - means[7])/(0.5*(stds[8] + stds[7])))
                    cnr_ml2.append((means[9] - means[8])/(0.5*(stds[9] + stds[8])))
                except:
                    continue

        all_cnr_stats["biexp"][i].append(np.nanmean(cnr_si1)) 
        all_cnr_stats["biexp"][i].append(np.nanmean(cnr_si2)) 
        all_cnr_stats["biexp"][i].append(np.nanmean(cnr_ap1)) 
        all_cnr_stats["biexp"][i].append(np.nanmean(cnr_ap2)) 
        all_cnr_stats["biexp"][i].append(np.nanmean(cnr_ml1)) 
        all_cnr_stats["biexp"][i].append(np.nanmean(cnr_ml2)) 
        all_cnr_avgs["biexp"][i] = np.mean(all_cnr_stats["biexp"][i])

    img_type = "triexp"
    all_cnr_stats["triexp"] = []
    all_cnr_avgs["triexp"] = []
    for i in range(5):
        all_cnr_stats["triexp"].append([])
        all_cnr_avgs["triexp"].append([])
        cnr_si1 = []
        cnr_si2 = []
        cnr_ap1 = []
        cnr_ap2 = []
        cnr_ml1 = []
        cnr_ml2 = []

        for patient in patient_nums:
            for roi in ["r_par", "l_par"]:
                try:
                    means = all_stats[patient][roi][img_type][i][0]
                    stds = all_stats[patient][roi][img_type][i][1]
                    cnr_si1.append((means[2] - means[1])/(0.5*(stds[2] + stds[1])))
                    cnr_si2.append((means[3] - means[2])/(0.5*(stds[3] + stds[2])))
                    cnr_ap1.append((means[5] - means[4])/(0.5*(stds[5] + stds[4])))
                    cnr_ap2.append((means[6] - means[5])/(0.5*(stds[6] + stds[5])))
                    cnr_ml1.append((means[8] - means[7])/(0.5*(stds[8] + stds[7])))
                    cnr_ml2.append((means[9] - means[8])/(0.5*(stds[9] + stds[8])))
                except:
                    continue

        all_cnr_stats["triexp"][i].append(np.nanmean(cnr_si1)) 
        all_cnr_stats["triexp"][i].append(np.nanmean(cnr_si2)) 
        all_cnr_stats["triexp"][i].append(np.nanmean(cnr_ap1)) 
        all_cnr_stats["triexp"][i].append(np.nanmean(cnr_ap2)) 
        all_cnr_stats["triexp"][i].append(np.nanmean(cnr_ml1)) 
        all_cnr_stats["triexp"][i].append(np.nanmean(cnr_ml2)) 
        all_cnr_avgs["triexp"][i] = np.mean(all_cnr_stats["triexp"][i])

    img_type = "adcs"
    all_cnr_stats["adcs"] = []

    cnr_si1 = []
    cnr_si2 = []
    cnr_ap1 = []
    cnr_ap2 = []
    cnr_ml1 = []
    cnr_ml2 = []

    for patient in patient_nums:
        for roi in ["r_par", "l_par"]:
            try:
                means = all_stats[patient][roi][img_type][0]
                stds = all_stats[patient][roi][img_type][1]
                cnr_si1.append((means[2] - means[1])/(0.5*(stds[2] + stds[1])))
                cnr_si2.append((means[3] - means[2])/(0.5*(stds[3] + stds[2])))
                cnr_ap1.append((means[5] - means[4])/(0.5*(stds[5] + stds[4])))
                cnr_ap2.append((means[6] - means[5])/(0.5*(stds[6] + stds[5])))
                cnr_ml1.append((means[8] - means[7])/(0.5*(stds[8] + stds[7])))
                cnr_ml2.append((means[9] - means[8])/(0.5*(stds[9] + stds[8])))
            except:
                continue

    all_cnr_stats["adcs"].append(np.nanmean(cnr_si1)) 
    all_cnr_stats["adcs"].append(np.nanmean(cnr_si2)) 
    all_cnr_stats["adcs"].append(np.nanmean(cnr_ap1)) 
    all_cnr_stats["adcs"].append(np.nanmean(cnr_ap2)) 
    all_cnr_stats["adcs"].append(np.nanmean(cnr_ml1)) 
    all_cnr_stats["adcs"].append(np.nanmean(cnr_ml2)) 
    all_cnr_avgs["adcs"] = np.mean(all_cnr_stats["adcs"])


    img_type = "aucs"
    all_cnr_stats["aucs"] = []

    cnr_si1 = []
    cnr_si2 = []
    cnr_ap1 = []
    cnr_ap2 = []
    cnr_ml1 = []
    cnr_ml2 = []

    for patient in patient_nums:
        for roi in ["r_par", "l_par"]:
            try:
                means = all_stats[patient][roi][img_type][0]
                stds = all_stats[patient][roi][img_type][1]
                cnr_si1.append((means[2] - means[1])/(0.5*(stds[2] + stds[1])))
                cnr_si2.append((means[3] - means[2])/(0.5*(stds[3] + stds[2])))
                cnr_ap1.append((means[5] - means[4])/(0.5*(stds[5] + stds[4])))
                cnr_ap2.append((means[6] - means[5])/(0.5*(stds[6] + stds[5])))
                cnr_ml1.append((means[8] - means[7])/(0.5*(stds[8] + stds[7])))
                cnr_ml2.append((means[9] - means[8])/(0.5*(stds[9] + stds[8])))
            except:
                continue

    all_cnr_stats["aucs"].append(np.nanmean(cnr_si1)) 
    all_cnr_stats["aucs"].append(np.nanmean(cnr_si2)) 
    all_cnr_stats["aucs"].append(np.nanmean(cnr_ap1)) 
    all_cnr_stats["aucs"].append(np.nanmean(cnr_ap2)) 
    all_cnr_stats["aucs"].append(np.nanmean(cnr_ml1)) 
    all_cnr_stats["aucs"].append(np.nanmean(cnr_ml2)) 
    all_cnr_avgs["aucs"] = np.mean(all_cnr_stats["aucs"])

    img_type = "aucs_l"
    all_cnr_stats["aucs_l"] = []

    cnr_si1 = []
    cnr_si2 = []
    cnr_ap1 = []
    cnr_ap2 = []
    cnr_ml1 = []
    cnr_ml2 = []

    for patient in patient_nums:
        for roi in ["r_par", "l_par"]:
            try:
                means = all_stats[patient][roi][img_type][0]
                stds = all_stats[patient][roi][img_type][1]
                cnr_si1.append((means[2] - means[1])/(0.5*(stds[2] + stds[1])))
                cnr_si2.append((means[3] - means[2])/(0.5*(stds[3] + stds[2])))
                cnr_ap1.append((means[5] - means[4])/(0.5*(stds[5] + stds[4])))
                cnr_ap2.append((means[6] - means[5])/(0.5*(stds[6] + stds[5])))
                cnr_ml1.append((means[8] - means[7])/(0.5*(stds[8] + stds[7])))
                cnr_ml2.append((means[9] - means[8])/(0.5*(stds[9] + stds[8])))
            except:
                continue

    all_cnr_stats["aucs_l"].append(np.nanmean(cnr_si1)) 
    all_cnr_stats["aucs_l"].append(np.nanmean(cnr_si2)) 
    all_cnr_stats["aucs_l"].append(np.nanmean(cnr_ap1)) 
    all_cnr_stats["aucs_l"].append(np.nanmean(cnr_ap2)) 
    all_cnr_stats["aucs_l"].append(np.nanmean(cnr_ml1)) 
    all_cnr_stats["aucs_l"].append(np.nanmean(cnr_ml2)) 
    all_cnr_avgs["aucs_l"] = np.mean(all_cnr_stats["aucs_l"])

    img_type = "aucs_m"
    all_cnr_stats["aucs_m"] = []

    cnr_si1 = []
    cnr_si2 = []
    cnr_ap1 = []
    cnr_ap2 = []
    cnr_ml1 = []
    cnr_ml2 = []

    for patient in patient_nums:
        for roi in ["r_par", "l_par"]:
            try:
                means = all_stats[patient][roi][img_type][0]
                stds = all_stats[patient][roi][img_type][1]
                cnr_si1.append((means[2] - means[1])/(0.5*(stds[2] + stds[1])))
                cnr_si2.append((means[3] - means[2])/(0.5*(stds[3] + stds[2])))
                cnr_ap1.append((means[5] - means[4])/(0.5*(stds[5] + stds[4])))
                cnr_ap2.append((means[6] - means[5])/(0.5*(stds[6] + stds[5])))
                cnr_ml1.append((means[8] - means[7])/(0.5*(stds[8] + stds[7])))
                cnr_ml2.append((means[9] - means[8])/(0.5*(stds[9] + stds[8])))
            except:
                continue

    all_cnr_stats["aucs_m"].append(np.nanmean(cnr_si1)) 
    all_cnr_stats["aucs_m"].append(np.nanmean(cnr_si2)) 
    all_cnr_stats["aucs_m"].append(np.nanmean(cnr_ap1)) 
    all_cnr_stats["aucs_m"].append(np.nanmean(cnr_ap2)) 
    all_cnr_stats["aucs_m"].append(np.nanmean(cnr_ml1)) 
    all_cnr_stats["aucs_m"].append(np.nanmean(cnr_ml2)) 
    all_cnr_avgs["aucs_m"] = np.mean(all_cnr_stats["aucs_m"])

    img_type = "aucs_h"
    all_cnr_stats["aucs_h"] = []

    cnr_si1 = []
    cnr_si2 = []
    cnr_ap1 = []
    cnr_ap2 = []
    cnr_ml1 = []
    cnr_ml2 = []

    for patient in patient_nums:
        for roi in ["r_par", "l_par"]:
            try:
                means = all_stats[patient][roi][img_type][0]
                stds = all_stats[patient][roi][img_type][1]
                cnr_si1.append((means[2] - means[1])/(0.5*(stds[2] + stds[1])))
                cnr_si2.append((means[3] - means[2])/(0.5*(stds[3] + stds[2])))
                cnr_ap1.append((means[5] - means[4])/(0.5*(stds[5] + stds[4])))
                cnr_ap2.append((means[6] - means[5])/(0.5*(stds[6] + stds[5])))
                cnr_ml1.append((means[8] - means[7])/(0.5*(stds[8] + stds[7])))
                cnr_ml2.append((means[9] - means[8])/(0.5*(stds[9] + stds[8])))
            except:
                continue

    all_cnr_stats["aucs_h"].append(np.nanmean(cnr_si1)) 
    all_cnr_stats["aucs_h"].append(np.nanmean(cnr_si2)) 
    all_cnr_stats["aucs_h"].append(np.nanmean(cnr_ap1)) 
    all_cnr_stats["aucs_h"].append(np.nanmean(cnr_ap2)) 
    all_cnr_stats["aucs_h"].append(np.nanmean(cnr_ml1)) 
    all_cnr_stats["aucs_h"].append(np.nanmean(cnr_ml2)) 
    all_cnr_avgs["aucs_h"] = np.mean(all_cnr_stats["aucs_h"])

    img_type = "aucs_s0"
    all_cnr_stats["aucs_s0"] = []

    cnr_si1 = []
    cnr_si2 = []
    cnr_ap1 = []
    cnr_ap2 = []
    cnr_ml1 = []
    cnr_ml2 = []

    for patient in patient_nums:
        for roi in ["r_par", "l_par"]:
            try:
                means = all_stats[patient][roi][img_type][0]
                stds = all_stats[patient][roi][img_type][1]
                cnr_si1.append((means[2] - means[1])/(0.5*(stds[2] + stds[1])))
                cnr_si2.append((means[3] - means[2])/(0.5*(stds[3] + stds[2])))
                cnr_ap1.append((means[5] - means[4])/(0.5*(stds[5] + stds[4])))
                cnr_ap2.append((means[6] - means[5])/(0.5*(stds[6] + stds[5])))
                cnr_ml1.append((means[8] - means[7])/(0.5*(stds[8] + stds[7])))
                cnr_ml2.append((means[9] - means[8])/(0.5*(stds[9] + stds[8])))
            except:
                continue

    all_cnr_stats["aucs_s0"].append(np.nanmean(cnr_si1)) 
    all_cnr_stats["aucs_s0"].append(np.nanmean(cnr_si2)) 
    all_cnr_stats["aucs_s0"].append(np.nanmean(cnr_ap1)) 
    all_cnr_stats["aucs_s0"].append(np.nanmean(cnr_ap2)) 
    all_cnr_stats["aucs_s0"].append(np.nanmean(cnr_ml1)) 
    all_cnr_stats["aucs_s0"].append(np.nanmean(cnr_ml2)) 
    all_cnr_avgs["aucs_s0"] = np.mean(all_cnr_stats["aucs_s0"])

    img_type = "aucs_s0_l"
    all_cnr_stats["aucs_s0_l"] = []

    cnr_si1 = []
    cnr_si2 = []
    cnr_ap1 = []
    cnr_ap2 = []
    cnr_ml1 = []
    cnr_ml2 = []

    for patient in patient_nums:
        for roi in ["r_par", "l_par"]:
            try:
                means = all_stats[patient][roi][img_type][0]
                stds = all_stats[patient][roi][img_type][1]
                cnr_si1.append((means[2] - means[1])/(0.5*(stds[2] + stds[1])))
                cnr_si2.append((means[3] - means[2])/(0.5*(stds[3] + stds[2])))
                cnr_ap1.append((means[5] - means[4])/(0.5*(stds[5] + stds[4])))
                cnr_ap2.append((means[6] - means[5])/(0.5*(stds[6] + stds[5])))
                cnr_ml1.append((means[8] - means[7])/(0.5*(stds[8] + stds[7])))
                cnr_ml2.append((means[9] - means[8])/(0.5*(stds[9] + stds[8])))
            except:
                continue

    all_cnr_stats["aucs_s0_l"].append(np.nanmean(cnr_si1)) 
    all_cnr_stats["aucs_s0_l"].append(np.nanmean(cnr_si2)) 
    all_cnr_stats["aucs_s0_l"].append(np.nanmean(cnr_ap1)) 
    all_cnr_stats["aucs_s0_l"].append(np.nanmean(cnr_ap2)) 
    all_cnr_stats["aucs_s0_l"].append(np.nanmean(cnr_ml1)) 
    all_cnr_stats["aucs_s0_l"].append(np.nanmean(cnr_ml2)) 
    all_cnr_avgs["aucs_s0_l"] = np.mean(all_cnr_stats["aucs_s0_l"])

    img_type = "aucs_s0_m"
    all_cnr_stats["aucs_s0_m"] = []

    cnr_si1 = []
    cnr_si2 = []
    cnr_ap1 = []
    cnr_ap2 = []
    cnr_ml1 = []
    cnr_ml2 = []

    for patient in patient_nums:
        for roi in ["r_par", "l_par"]:
            try:
                means = all_stats[patient][roi][img_type][0]
                stds = all_stats[patient][roi][img_type][1]
                cnr_si1.append((means[2] - means[1])/(0.5*(stds[2] + stds[1])))
                cnr_si2.append((means[3] - means[2])/(0.5*(stds[3] + stds[2])))
                cnr_ap1.append((means[5] - means[4])/(0.5*(stds[5] + stds[4])))
                cnr_ap2.append((means[6] - means[5])/(0.5*(stds[6] + stds[5])))
                cnr_ml1.append((means[8] - means[7])/(0.5*(stds[8] + stds[7])))
                cnr_ml2.append((means[9] - means[8])/(0.5*(stds[9] + stds[8])))
            except:
                continue

    all_cnr_stats["aucs_s0_m"].append(np.nanmean(cnr_si1)) 
    all_cnr_stats["aucs_s0_m"].append(np.nanmean(cnr_si2)) 
    all_cnr_stats["aucs_s0_m"].append(np.nanmean(cnr_ap1)) 
    all_cnr_stats["aucs_s0_m"].append(np.nanmean(cnr_ap2)) 
    all_cnr_stats["aucs_s0_m"].append(np.nanmean(cnr_ml1)) 
    all_cnr_stats["aucs_s0_m"].append(np.nanmean(cnr_ml2)) 
    all_cnr_avgs["aucs_s0_m"] = np.mean(all_cnr_stats["aucs_s0_m"])

    img_type = "aucs_s0_h"
    all_cnr_stats["aucs_s0_h"] = []

    cnr_si1 = []
    cnr_si2 = []
    cnr_ap1 = []
    cnr_ap2 = []
    cnr_ml1 = []
    cnr_ml2 = []

    for patient in patient_nums:
        for roi in ["r_par", "l_par"]:
            try:
                means = all_stats[patient][roi][img_type][0]
                stds = all_stats[patient][roi][img_type][1]
                cnr_si1.append((means[2] - means[1])/(0.5*(stds[2] + stds[1])))
                cnr_si2.append((means[3] - means[2])/(0.5*(stds[3] + stds[2])))
                cnr_ap1.append((means[5] - means[4])/(0.5*(stds[5] + stds[4])))
                cnr_ap2.append((means[6] - means[5])/(0.5*(stds[6] + stds[5])))
                cnr_ml1.append((means[8] - means[7])/(0.5*(stds[8] + stds[7])))
                cnr_ml2.append((means[9] - means[8])/(0.5*(stds[9] + stds[8])))
            except:
                continue

    all_cnr_stats["aucs_s0_h"].append(np.nanmean(cnr_si1)) 
    all_cnr_stats["aucs_s0_h"].append(np.nanmean(cnr_si2)) 
    all_cnr_stats["aucs_s0_h"].append(np.nanmean(cnr_ap1)) 
    all_cnr_stats["aucs_s0_h"].append(np.nanmean(cnr_ap2)) 
    all_cnr_stats["aucs_s0_h"].append(np.nanmean(cnr_ml1)) 
    all_cnr_stats["aucs_s0_h"].append(np.nanmean(cnr_ml2)) 
    all_cnr_avgs["aucs_s0_h"] = np.mean(all_cnr_stats["aucs_s0_h"])

    save_path = os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", str("cnr_roi_stats_" + scan_type + "_" + str(deblurred) + ".txt"))
    with open(save_path, "wb") as fp:
        pickle.dump([all_cnr_stats, all_cnr_avgs], fp)

    return all_cnr_avgs

def get_param_statistics(predict_s0=False):
    pre_stats_deblurred, dose_stats = get_all_parotid_param_and_dose_stats(deblurred=True, scan_type="pre", predict_s0=predict_s0)
    post_stats_deblurred, dose_stats = get_all_parotid_param_and_dose_stats(deblurred=True, scan_type="post", predict_s0=predict_s0)

    pre_stats_orig, dose_stats = get_all_parotid_param_and_dose_stats(deblurred=False, scan_type="pre", predict_s0=predict_s0)
    post_stats_orig, dose_stats = get_all_parotid_param_and_dose_stats(deblurred=False, scan_type="post", predict_s0=predict_s0)

    with open(os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", str("param_roi_stats_" + "pre" + "_" + "True" + ".txt")), "rb") as fp:
        pre_stats_deblurred = pickle.load(fp)

    with open(os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", str("param_roi_stats_" + "post" + "_" + "True" + ".txt")), "rb") as fp:
        post_stats_deblurred = pickle.load(fp)

    with open(os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", str("param_roi_stats_" + "pre" + "_" + "False" + ".txt")), "rb") as fp:
        pre_stats_orig = pickle.load(fp)

    with open(os.path.join("/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/fitting_ivim_deep", str("param_roi_stats_" + "post" + "_" + "False" + ".txt")), "rb") as fp:
        post_stats_orig = pickle.load(fp)
    
    #see if statistical difference between deblurred and blurred parameters
    biexp_ps = []
    import scipy.stats as stats
    for i in range(3):
        all_stats_deblurred = []
        all_stats_orig = []
        for patient in pre_stats_deblurred:
            if patient == "all":
                continue
            for roi in pre_stats_deblurred[patient]:
                all_stats_deblurred.append(pre_stats_deblurred[patient][roi]["biexps"][i][0][0])
                all_stats_orig.append(pre_stats_orig[patient][roi]["biexps"][i][0][0])

                #and post rt values
                try:
                    all_stats_deblurred.append(post_stats_deblurred[patient][roi]["biexps"][i][0][0])
                    all_stats_orig.append(post_stats_orig[patient][roi]["biexps"][i][0][0])
                except:
                    continue

        avg1 = np.mean(all_stats_deblurred)
        avg2 = np.mean(all_stats_orig)
        t, p = stats.ttest_rel(all_stats_deblurred, all_stats_orig)
        biexp_ps.append(p)

    triexp_ps = []
    import scipy.stats as stats
    for i in range(5):
        all_stats_deblurred = []
        all_stats_orig = []
        for patient in pre_stats_deblurred:
            if patient == "all":
                continue
            for roi in pre_stats_deblurred[patient]:
                all_stats_deblurred.append(pre_stats_deblurred[patient][roi]["triexp"][i][0][0])
                all_stats_orig.append(pre_stats_orig[patient][roi]["triexp"][i][0][0])
                try:
                    #and post rt values
                    all_stats_deblurred.append(post_stats_deblurred[patient][roi]["triexp"][i][0][0])
                    all_stats_orig.append(post_stats_orig[patient][roi]["triexp"][i][0][0])
                except:
                    continue

        t, p = stats.ttest_rel(all_stats_deblurred, all_stats_orig)
        triexp_ps.append(p)

    all_stats_deblurred = []
    all_stats_orig = []
    for patient in pre_stats_deblurred:
        if patient == "all":
                continue
        for roi in pre_stats_deblurred[patient]:
            all_stats_deblurred.append(pre_stats_deblurred[patient][roi]["adcs"][0][0])
            all_stats_orig.append(pre_stats_orig[patient][roi]["adcs"][0][0])

            #and post rt values
            try:
                all_stats_deblurred.append(post_stats_deblurred[patient][roi]["adcs"][0][0])
                all_stats_orig.append(post_stats_orig[patient][roi]["adcs"][0][0])
            except:
                continue
    t, adc_p = stats.ttest_rel(all_stats_deblurred, all_stats_orig)

    auc_ps = []
    for auc in ["aucs_s0", "aucs_s0_l", "aucs_s0_m", "aucs_s0_h"]:
        all_stats_deblurred = []
        all_stats_orig = []
        for patient in pre_stats_deblurred:
            if patient == "all":
                continue
            for roi in pre_stats_deblurred[patient]:
                all_stats_deblurred.append(pre_stats_deblurred[patient][roi][auc][0][0])
                all_stats_orig.append(pre_stats_orig[patient][roi][auc][0][0])

                #and post rt values
                try:
                    all_stats_deblurred.append(post_stats_deblurred[patient][roi][auc][0][0])
                    all_stats_orig.append(post_stats_orig[patient][roi][auc][0][0])
                except:
                    continue

        t, auc_p = stats.ttest_rel(all_stats_deblurred, all_stats_orig)
        auc_ps.append(auc_p)

    
    

    print("")
    return



if __name__ == "__main__":
    #get_param_statistics()

    #view_blur_deblur_and_conv()
    # plot_avg_param_subsegs(deblurred=True)
    # plot_avg_param_subsegs(deblurred=False)
    #clean_parameter_predictions()

    # train_adc_model(predict_s0=False)
    # train_triexp_model(predict_s0=False)
    # train_biexp_model(predict_s0=False)

    # param_stats, dose_stats = get_all_parotid_param_and_dose_stats(deblurred=True, scan_type="pre")
    # param_stats, dose_stats = get_all_parotid_param_and_dose_stats(deblurred=False, scan_type="pre")
    # param_stats, dose_stats = get_all_parotid_param_and_dose_stats(deblurred=True, scan_type="post")
    # param_stats, dose_stats = get_all_parotid_param_and_dose_stats(deblurred=False, scan_type="post")

    # get_all_parotid_histogram_stats(deblurred=True, scan_type="pre")
    # get_all_parotid_histogram_stats(deblurred=False, scan_type="pre")
    # get_all_parotid_histogram_stats(deblurred=True, scan_type="post")
    # get_all_parotid_histogram_stats(deblurred=False, scan_type="post")
    
    # print("to rel")
    dose_to_ivim_change_analysis_rel(deblurred=True)
    dose_to_ivim_change_analysis_rel(deblurred=False)



    # cnrs_deblurred= get_cnrs(deblurred=True)
    # cnrs = get_cnrs(deblurred=False)



    #get_parameter_svd_variance(deblurred=True)
    # get_parameter_svd_variance(deblurred=False)

    # plot_ivim_hists_blur_and_deblur()



    patient_nums = ["10", "11", "12", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']
    #patient_nums = ['17', '18', '19', '20', '21', '22', '23']
    #patient_nums = ["10", "11", "12", "13", "16"]

    scan_type = "pre"
    # #scan_type = "post"
    # predict_all_biexp_imgs(patient_nums, scan_type, deblurred=True, predict_s0=True)
    # predict_all_biexp_imgs(patient_nums,scan_type, deblurred=False, predict_s0=True)


    # predict_all_triexp_imgs(patient_nums,scan_type, deblurred=True, predict_s0=True)
    # predict_all_triexp_imgs(patient_nums, scan_type,deblurred=False, predict_s0=True)
    
    
    # predict_all_adc_imgs(patient_nums, scan_type,deblurred=True, predict_s0=True)
    # predict_all_adc_imgs(patient_nums,scan_type, deblurred=False, predict_s0=True)

    # predict_all_auc_imgs(patient_nums,scan_type, deblurred=True, predict_s0=True)
    # predict_all_auc_imgs(patient_nums,scan_type, deblurred=False, predict_s0=True)

    # predict_all_auc_imgs(patient_nums,scan_type, deblurred=True, predict_s0=False)
    # predict_all_auc_imgs(patient_nums,scan_type, deblurred=False, predict_s0=False)

   
    #view_all_predicted_imgs(patient_nums, scan_type, deblurred=True)
    
    
    #make_curvature_plot(deblurred=True)

    # get_triexp_param_heatmap()
    #get_biexp_param_heatmap(deblurred=False)
    #get_biexp_param_heatmap(deblurred=True)
    # train_adc_model()
    # train_triexp_model()
    # train_biexp_model()
    
    
    # train_segmented_D_model(deblurred=True)



    
    # train_segmented_D_model()
    


