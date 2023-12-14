import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import math
import SimpleITK as sitk
import numpy as np
from blind_deconv import plot_3d_image
from pytorch_msssim import ssim, ms_ssim
from scipy.ndimage import label

mse = torch.nn.MSELoss()
mae = nn.L1Loss()

class Kernel_Loss(nn.Module):
    def __init__(self):
        super(Kernel_Loss, self).__init__()
    def forward(self, k,y):
        #l1_loss = nn.L1Loss()(k,y)
        mse_loss = nn.MSELoss()(k,y) * 100000
        return mse_loss    

class Deconv_Loss(nn.Module):
    def __init__(self, device, max_noise):
        super(Deconv_Loss, self).__init__()
        self.tv_weight = 0.01
        self.kernel_weight = 1
        self.noise_max = max_noise
        self.signal_decay_weight = 1
        self.noise_weight = 0.1
        self.dose_weight = 10e-5
        self.curvature_weight = 0.01
        self.centre_weight = 1
        self.b_vals = torch.tensor([0,20,30,40,50,60,70,80,90,100,120,150,250,400,800,1000]).to(device)
        self.b_diffs = self.b_vals[1:] - self.b_vals[:-1]
    def forward(self, prediction, target, kernel, deblurred_prediction, noise, step):
        if step < 2000:

            # dose_penalty = torch.abs(torch.sum(target)-torch.sum(prediction))

            fidelity_term = mse(prediction, target)# + mae(prediction, target) 
            loss_kernel = torch.norm(kernel[kernel > 0.7], p=2) 


            noise_term = noise[noise > self.noise_max]
            if noise_term.numel() == 0:
                noise_term = 0
            else:
                noise_term = torch.nanmean(torch.abs((noise_term)))
            #get connectivity term
            


            loss = fidelity_term + (self.kernel_weight * (loss_kernel))  + self.noise_weight * noise_term 

            if step < 2000:

                _, _, k_z, k_y, k_x = kernel.shape
                grid_z, grid_y, grid_x = torch.meshgrid(torch.arange(k_z), torch.arange(k_y), torch.arange(k_x))
                grid_z = grid_z.to(kernel.device)
                grid_y = grid_y.to(kernel.device)
                grid_x = grid_x.to(kernel.device)
                distance_z = torch.abs(grid_z -int(k_z/2))
                distance_y = torch.abs(grid_y - int(k_y/2))
                distance_x = torch.abs(grid_x - int(k_x/2))
                #dont penalize at all if near centre
                distance_z[distance_z < 2] = 0 
                distance_x[distance_x < 2] = 0 
                distance_y[distance_y < 2] = 0 
                centre_loss =kernel[0,0,:,:,:] * (distance_z + distance_y + distance_x)
                centre_loss = centre_loss.sum()    
                loss = loss + (centre_loss * self.centre_weight )

            if step % 25 == 0:
                if (self.kernel_weight * loss_kernel) > 0.2*fidelity_term.item():
                    self.kernel_weight = 0.02 * fidelity_term.item() / loss_kernel.item()   #make weight 1% of fidelity term
                if step < 2000 and (centre_loss * self.centre_weight) > 0.1*fidelity_term.item():
                    self.centre_weight = 0.01 * fidelity_term.item() / centre_loss.item()   #make weight 1% of fidelity term    
                # if (self.signal_decay_weight * loss_signal_decay) > 0.1*fidelity_term.item():
                #     self.signal_decay_weight = 0.01 * fidelity_term.item() / loss_signal_decay.item()   #make weight 1% of fidelity term
                if (self.noise_weight * noise_term) > 0.5*fidelity_term.item():
                    self.noise_weight = 0.1 * fidelity_term.item() / noise_term.item()   #make weight 1% of fidelity term

            if step % 100 == 0:
                print(f"fidelity loss: {round(fidelity_term.item(),8)}|| kernel loss: {round(self.kernel_weight*loss_kernel.item(),8)}")# || curvature loss: {round(self.curvature_weight*loss_curvature.item(),8)}")# || dose loss: {round(self.dose_weight*dose_penalty.item(),8)} ")


        elif step < 3500:   #introduce tv loss and make kernel loss weaker
            fidelity_term =  mse(prediction, target)#+ mae(prediction, target)#1-ms_ssim(pad_if_small(prediction), pad_if_small(target), data_range=1)#mse(prediction, target) 
            norm_penalty =torch.norm(kernel[kernel > 0.7], p=2)
            #tv_penalty = tv_kernel * self.tv_weight
            loss_kernel =  norm_penalty 

            noise_term = noise[noise > self.noise_max]

            if noise_term.numel() == 0:
                noise_term = 0
            else:
                noise_term = torch.nanmean(torch.abs((noise_term)))

         
            loss = fidelity_term + (self.kernel_weight * (loss_kernel)) + self.noise_weight * noise_term#+ (self.curvature_weight * loss_curvature) #+ (self.dose_weight * dose_penalty)
            
            
            
            _, _, k_z, k_y, k_x = kernel.shape
            grid_z, grid_y, grid_x = torch.meshgrid(torch.arange(k_z), torch.arange(k_y), torch.arange(k_x))
            grid_z = grid_z.to(kernel.device)
            grid_y = grid_y.to(kernel.device)
            grid_x = grid_x.to(kernel.device)
            distance_z = torch.abs(grid_z -int(k_z/2))
            distance_y = torch.abs(grid_y - int(k_y/2))
            distance_x = torch.abs(grid_x - int(k_x/2))
            #dont penalize at all if near centre
            distance_z[distance_z < 2] = 0 
            distance_x[distance_x < 2] = 0 
            distance_y[distance_y < 2] = 0 
            centre_loss =kernel[0,0,:,:,:] * (distance_z + distance_y + distance_x)
            centre_loss = centre_loss.sum()    
            loss = loss + (centre_loss * self.centre_weight )



            if step % 25 == 0:
                if (self.kernel_weight * loss_kernel) > 0.2*fidelity_term.item():
                    self.kernel_weight = 0.01 * fidelity_term.item() / loss_kernel.item()   #make weight 1% of fidelity term
                if (self.noise_weight * noise_term) > 0.5*fidelity_term.item():
                    self.noise_weight = 0.1 * fidelity_term.item() / noise_term.item()   #make weight 1% of fidelity term
                if (centre_loss * self.centre_weight) > 0.1*fidelity_term.item():
                    self.centre_weight = 0.01 * fidelity_term.item() / centre_loss.item()   #make weight 1% of fidelity term   
            if step % 100 == 0:
                print(f"fidelity loss: {round(fidelity_term.item(),8)}|| kernel loss: {round(self.kernel_weight*loss_kernel.item(),8)}")# || tv loss: {round(self.tv_weight*tv_loss.item(),8)}")
            #print(f"fidelity loss: {round(fidelity_term.item(),8)}|| kernel loss: {round(self.kernel_weight*loss_kernel.item(),8)} || curvature loss: {round(self.curvature_weight*loss_curvature.item(),8)} || dose loss: {round(self.dose_weight*dose_penalty.item(),8)} || tv loss: {round(self.tv_weight*tv_loss.item(),8)}")


        else:   #introduce tv loss and make kernel loss weaker
            fidelity_term = 1-ms_ssim(pad_if_small(prediction), pad_if_small(target), data_range=1)#mse(prediction, target) + mae(prediction, target)
            
            norm_penalty =torch.norm(kernel[kernel > 0.7], p=2) + 0.01*torch.norm(kernel[kernel < 0.01], p=2)#0.01*torch.norm(kernel[kernel < 0.002], p=2)
            #tv_penalty = tv_kernel * self.tv_weight
            loss_kernel =  norm_penalty 

            
            # diffs = deblurred_prediction[:,1:,:,:,:] - deblurred_prediction[:,:-1,:,:,:]
            
            # decreasing_with_b_term = torch.nanmean(torch.clamp(diffs, min=0))

            # #also want a term so that the gradient is always increasing (exponential decay) (second derivative positive)
            # grad = diffs / self.b_diffs.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
            # curvature = (grad[:,1:,:,:,:] - grad[:,:-1,:,:,:]) / self.b_diffs[1:].unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
            
            # if curvature[curvature < 0] == []:
            #     curvature = 0
            # else:         
            #     curvature = torch.nanmean(torch.abs(curvature[curvature < 0]))   #penalize if curvature is < 0 
            
            #loss_signal_decay =  decreasing_with_b_term #+ curvature


            grad_x = torch.abs(deblurred_prediction[:,:,:,:,:-1] - deblurred_prediction[:,:,:,:,1:])
            grad_y = torch.abs(deblurred_prediction[:,:,:,:-1,:] - deblurred_prediction[:,:,:,1:,:])
            grad_z = torch.abs(deblurred_prediction[:,:,:-1,:,:] - deblurred_prediction[:,:,1:,:,:])
            tv_loss = (torch.mean(grad_x) + torch.mean(grad_y) + torch.mean(grad_z))

            noise_term = noise[noise > self.noise_max]
            if noise_term.numel() == 0:
                noise_term = 0
            else:
                noise_term = torch.nanmean(torch.abs((noise_term)))
            loss = fidelity_term + (self.kernel_weight * (loss_kernel))  + self.noise_weight * noise_term + self.tv_weight * tv_loss



            _, _, k_z, k_y, k_x = kernel.shape
            grid_z, grid_y, grid_x = torch.meshgrid(torch.arange(k_z), torch.arange(k_y), torch.arange(k_x))
            grid_z = grid_z.to(kernel.device)
            grid_y = grid_y.to(kernel.device)
            grid_x = grid_x.to(kernel.device)
            distance_z = torch.abs(grid_z -int(k_z/2))
            distance_y = torch.abs(grid_y - int(k_y/2))
            distance_x = torch.abs(grid_x - int(k_x/2))
            #dont penalize at all if near centre
            distance_z[distance_z < 2] = 0 
            distance_x[distance_x < 2] = 0 
            distance_y[distance_y < 2] = 0 
            centre_loss =kernel[0,0,:,:,:] * (distance_z + distance_y + distance_x)
            centre_loss = centre_loss.sum()    
            loss = loss + (centre_loss * self.centre_weight )



            if step % 25 == 0:
                if (self.kernel_weight * loss_kernel) > 0.2*fidelity_term.item():
                    self.kernel_weight = 0.01 * fidelity_term.item() / loss_kernel.item()   #make weight 1% of fidelity term
                if (self.noise_weight * noise_term) > 0.5*fidelity_term.item():
                    self.noise_weight = 0.1 * fidelity_term.item() / noise_term.item()   #make weight 1% of fidelity term
                if (self.tv_weight * tv_loss) > 0.01*fidelity_term.item():
                    self.tv_weight = 0.001 * fidelity_term.item() / tv_loss.item()   #make weight 1% of fidelity term
                if (centre_loss * self.centre_weight) > 0.01*fidelity_term.item():
                    self.centre_weight = 0.001 * fidelity_term.item() / centre_loss.item()   #make weight 1% of fidelity term   
        if loss.item == torch.nan:
            raise Exception("nan loss")
        return loss#, fidelity_term, norm_term, tv_loss, tv_loss_mask, loss_kernel
    
class Deconv_Loss_Small(nn.Module):
    def __init__(self,  device):
        super(Deconv_Loss_Small, self).__init__()

        self.kernel_weight = 10e-1

        self.centre_weight = 1
        self.curvature_weight = 0.01
        self.signal_decay_weight = 1
        self.b_vals = torch.tensor([0,20,30,40,50,60,70,80,90,100,120,150,250,400,800,1000]).to(device)
        self.b_diffs = (self.b_vals[1:] - self.b_vals[:-1]).to(device)
    def forward(self, prediction, target, kernel, deblurred_prediction, step):
  

        fidelity_term = mse(prediction, target)#+ mae(prediction, target)

        loss_kernel =  torch.norm(kernel[kernel > 0.7], p=2) #+ 0.001*torch.norm(kernel[kernel < 0.001], p=2)
    
        loss = fidelity_term + (self.kernel_weight * (loss_kernel)) #self.signal_decay_weight * loss_signal_decay

        _, _, k_z, k_y, k_x = kernel.shape
        grid_z, grid_y, grid_x = torch.meshgrid(torch.arange(k_z), torch.arange(k_y), torch.arange(k_x))
        grid_z = grid_z.to(kernel.device)
        grid_y = grid_y.to(kernel.device)
        grid_x = grid_x.to(kernel.device)
        distance_z = torch.abs(grid_z -int(k_z/2))
        distance_y = torch.abs(grid_y - int(k_y/2))
        distance_x = torch.abs(grid_x - int(k_x/2))
        distance_z[distance_z < 2] = 0 
        distance_x[distance_x < 2] = 0 
        distance_y[distance_y < 2] = 0 
        centre_loss = kernel[0,0,:,:,:] * (distance_z + distance_y + distance_x)
        centre_loss = centre_loss.sum()

        loss = loss #+ (centre_loss * self.centre_weight )

        if step % 25 == 0:
            if (self.kernel_weight * loss_kernel) > 0.3*fidelity_term.item():
                self.kernel_weight = 0.05 * fidelity_term.item() / loss_kernel.item()   #make weight 1% of fidelity term

            if (centre_loss * self.centre_weight) > 0.1*fidelity_term.item():
                self.centre_weight = 0.01 * fidelity_term.item() / centre_loss.item()   #make weight 1% of fidelity term    


        if step % 100 == 0:
            print(f"fidelity loss: {round(fidelity_term.item(),8)}|| kernel loss: {round(self.kernel_weight*loss_kernel.item(),8)} || Center loss: {round(self.centre_weight*centre_loss.item(),8)}")
        return loss
  

def get_joint_entropy_penalty(x,y):
    #here implement the joint entropy loss term of the pet prediction and the ct texture map (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6853071/)
    #will include the high region, where psma uptake concentrates, as well as the low region which is the rest of the body where levels are low. 
    # This makes the probability distributions less concentrated in a single region
    if x.shape[-1] < 161:
        x = pad_if_small(x)
        y = pad_if_small(y)
    
    #m = 100#math.ceil(math.log2(num_voxels)+1)    #use sturges rule to determine the number of bins to use for discretization of both images

    #flatten the tensors
    # x = x.view(-1)
    # y = y.view(-1)
    # x_high = x[x > 0.15]    
    # y_high = y[x > 0.15]

    # x_low = x[torch.logical_and(x > 0.01, x < 0.15)]
    # y_low = y[torch.logical_and(x > 0.01, x < 0.15)]

    # #concatenate the arrays along last axis to compute the 2d histogram
    # combined_tensor_high = torch.cat((x_high.unsqueeze(-1), y_high.unsqueeze(-1)), dim=-1)
    # hist_high = torch.histogramdd(combined_tensor_high, bins=m)
    # bins_u_high, bins_v_high = hist_high.bin_edges
    # hist_high = hist_high.hist / torch.sum(hist_high.hist)

    # je_high = -torch.sum(hist_high * torch.log2(hist_high+1e-10))
    

    # #concatenate the arrays along last axis to compute the 2d histogram
    # combined_tensor_low = torch.cat((x_low.unsqueeze(-1), y_low.unsqueeze(-1)), dim=-1)
    # hist_low = torch.histogramdd(combined_tensor_low, bins=m)
    # bins_u_low, bins_v_low = hist_low.bin_edges
    # hist_low = hist_low.hist / torch.sum(hist_low.hist)

    # je_low = -torch.sum(hist_low * torch.log2(hist_low+1e-10))

    # plt.imshow(hist_low.detach().numpy())
    # plt.show(block=True)
    # plt.imshow(hist_high.detach().numpy())
    # plt.show(block=True)
    x = x.double()
    y = y.double()
    # x = (x - torch.min(x)) / (torch.max(x)-torch.min(x))
    # x_high = x.clone()
    # x_high[x < 0.15] = 0

    je = 1 - ms_ssim(x, y, data_range=1)
    x = x.float()
    y = y.float()

    return je

def pad_if_small(image):
    _,_,z_,n,_ = image.size()
    if n > 160:
        return image
    padding_n = (162-n+1) // 2

    padded_img = F.pad(image, (padding_n, padding_n,padding_n, padding_n)) #padding counts from last index ("to pad the last 2 dimensions of the input tensor, then use (padding_left,padding_right,(padding_left,padding_right, padding_top,padding_bottom)padding_top,padding_bottom))
    return padded_img