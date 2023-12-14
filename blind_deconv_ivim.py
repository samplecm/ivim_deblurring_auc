import os
import numpy as np
import torch
import torch.optim
import SimpleITK as sitk
import pickle
from main_importance_comparisons import plot_3d_image
import glob
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import model_deconv
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing, binary_fill_holes, label, zoom 
from utils import *
import loss_functions_ivim
import model_deconv_ivim
import torch.nn.functional as F
import radiomics
import time
import six 
import cv2
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as shifter
from copy import deepcopy

torch.cuda.empty_cache()

def main(patient_list, scan_type="post"):
    data_folder = os.path.join(os.getcwd(), "data_mri")
    patient_nums = os.listdir(data_folder)
    patient_nums.sort()
    for patient_num in patient_list:
        

        img_series_path = os.path.join(os.getcwd(), "data_mri")
        img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + scan_type + "diffusion_imgs"))
        with open(img_series_path, "rb") as fp:
            img_dict_full = pickle.load(fp)
        img_dict = img_dict_full["diffusion"]
        with open(os.path.join(data_folder, str("sivim" + patient_num + "_" + scan_type + "_MR_mask_dict")), "rb") as fp:
            mask_dict = pickle.load(fp)

        # #now do with right par

        print(f"Loading r par data for for {patient_num}...")
        img, mask_dict, coords_array, b_vals = prepare_diff_images(img_dict, mask_dict,  patient_num)
        z_min = 1000
        z_max = -1000
        x_min = 1000
        x_max = -1000
        y_min = 1000
        y_max = -1000
        for structure in mask_dict:
            if "par" not in structure.lower() or "l" in structure.lower():
                continue

            mask = mask_dict[structure].whole_roi_masks_resampled
            z_min_mask, z_max_mask = np.min(np.where(mask)[0]),  np.max(np.where(mask)[0]) #[int(pet_img.shape[0]*0.3), int(pet_img.shape[0]*0.7)]
            if z_min_mask < z_min:
                z_min = z_min_mask
            if z_max_mask > z_max:
                z_max = z_max_mask  

            y_min_mask, y_max_mask = np.min(np.where(mask)[1]),  np.max(np.where(mask)[1]) #[int(pet_img.shape[0]*0.3), int(pet_img.shape[0]*0.7)]
            if y_min_mask < y_min:
                y_min = y_min_mask
            if y_max_mask > y_max:
                y_max = y_max_mask 

            x_min_mask, x_max_mask = np.min(np.where(mask)[2]),  np.max(np.where(mask)[2]) #[int(pet_img.shape[0]*0.3), int(pet_img.shape[0]*0.7)]
            if x_min_mask < x_min:
                x_min = x_min_mask
            if x_max_mask > x_max:
                x_max = x_max_mask 

            if (y_max - y_min) % 2 == 1: 
                y_max += 1
            if (x_max - x_min) % 2 == 1:
                x_max += 1
            delta = max(y_max - y_min, x_max-x_min) + 10
            if (delta) % 2 == 1:
                delta -= 1
            if (z_max - z_min ) % 2 == 1:
                z_min -= 1    

            pad_y = int((delta - (y_max-y_min)) / 2)
            pad_x = int((delta - (x_max-x_min)) / 2)
            mask_dict[structure].mask_deconv = mask[z_min-4:z_max+4,y_min-pad_y:y_max+pad_y, x_min-pad_x:x_max+pad_x]
            break
        
        #if structure not found, skip to next patient
        if z_min == 1000:
            print(f"Structure not found.. skipping patient {patient_num}")
            with open(os.path.join(data_folder, str("sivim" + patient_num + "_" + scan_type + "_MR_mask_dict")), "wb") as fp:
                pickle.dump(mask_dict, fp)

            with open(img_series_path, "wb") as fp:
                pickle.dump(img_dict_full, fp)
        else:

            img = img[:,z_min-4:z_max+4,y_min-pad_y:y_max+pad_y, x_min-pad_x:x_max+pad_x]
            img_dict.deconv_coords_r_par = coords_array[:,z_min-4:z_max+4,y_min-pad_y:y_max+pad_y, x_min-pad_x:x_max+pad_x]


            prediction_blurred, prediction_deblurred, kernel, img_orig, noise = blind_deconvolution(img, patient_num, load_models=False, side="right", scan_type=scan_type)
            img_dict.deconv_array_r_par = prediction_deblurred
            img_dict.deconv_array_blurred_r_par = prediction_blurred
            img_dict.img_r_par = img
            img_dict.kernel_r_par = kernel
            #img_dict.img_r_par = img_orig
            img_dict.noise_r_par = noise


            with open(img_series_path, "wb") as fp:
                pickle.dump(img_dict_full, fp)


        #do for left par
        print(f"Loading l par data for for {patient_num}...")
        img, mask_dict, coords_array, b_vals = prepare_diff_images(img_dict, mask_dict, patient_num)
        z_min = 1000
        z_max = -1000
        x_min = 1000
        x_max = -1000
        y_min = 1000
        y_max = -1000
        for structure in mask_dict:
            if "par" not in structure.lower() or "l" not in structure.lower(): #if not l par
                continue

            mask = mask_dict[structure].whole_roi_masks_resampled
            z_min_mask, z_max_mask = np.min(np.where(mask)[0]),  np.max(np.where(mask)[0]) #[int(pet_img.shape[0]*0.3), int(pet_img.shape[0]*0.7)]
            if z_min_mask < z_min:
                z_min = z_min_mask
            if z_max_mask > z_max:
                z_max = z_max_mask  

            y_min_mask, y_max_mask = np.min(np.where(mask)[1]),  np.max(np.where(mask)[1]) #[int(pet_img.shape[0]*0.3), int(pet_img.shape[0]*0.7)]
            if y_min_mask < y_min:
                y_min = y_min_mask
            if y_max_mask > y_max:
                y_max = y_max_mask 

            x_min_mask, x_max_mask = np.min(np.where(mask)[2]),  np.max(np.where(mask)[2]) #[int(pet_img.shape[0]*0.3), int(pet_img.shape[0]*0.7)]
            if x_min_mask < x_min:
                x_min = x_min_mask
            if x_max_mask > x_max:
                x_max = x_max_mask 
            if (y_max - y_min) % 2 == 1: 
                y_max += 1
            if (x_max - x_min) % 2 == 1:
                x_max += 1
            delta = max(y_max - y_min, x_max-x_min) + 10
            if (delta) % 2 == 1:
                delta -= 1
            if (z_max - z_min ) % 2 == 1:
                z_min -= 1    

            pad_y = int((delta - (y_max-y_min)) / 2)
            pad_x = int((delta - (x_max-x_min)) / 2)
            mask_dict[structure].mask_deconv = mask[z_min-4:z_max+4,y_min-pad_y:y_max+pad_y, x_min-pad_x:x_max+pad_x]
            break
        #if structure not found
        if z_min == 1000:
            print(f"Structure not found.. skipping patient {patient_num}")
            with open(os.path.join(data_folder, str("sivim" + patient_num + "_" + scan_type + "_MR_mask_dict")), "wb") as fp:
                pickle.dump(mask_dict, fp)

            with open(img_series_path, "wb") as fp:
                pickle.dump(img_dict_full, fp)
        else:  
             
            img = img[:,z_min-4:z_max+4,y_min-pad_y:y_max+pad_y, x_min-pad_x:x_max+pad_x]
            img_dict.deconv_coords_l_par = coords_array[:,z_min-4:z_max+4,y_min-pad_y:y_max+pad_y, x_min-pad_x:x_max+pad_x]


            prediction_blurred, prediction_deblurred, kernel, img_orig, noise = blind_deconvolution(img, patient_num, load_models=False, side="left", scan_type=scan_type)
            img_dict.deconv_array_l_par = prediction_deblurred
            img_dict.deconv_array_blurred_l_par = prediction_blurred
            img_dict.img_l_par = img
            img_dict.kernel_l_par = kernel
            #img_dict.img_l_par = img_orig
            img_dict.b_values = b_vals
            img_dict.noise_l_par = noise



        # #now do for cord
        # print(f"Loading cord data for for {patient_num}...")
        # img, mask_dict, coords_array, b_vals = prepare_diff_images(img_dict, mask_dict, patient_num)
        # z_min = 1000
        # z_max = -1000
        # x_min = 1000
        # x_max = -1000
        # y_min = 1000
        # y_max = -1000
        # for structure in mask_dict:
        #     if "cord" not in structure.lower(): #if not l par
        #         continue

        #     mask = mask_dict[structure].whole_roi_masks_resampled
            
        #     z_min_mask, z_max_mask = np.min(np.where(mask)[0]),  np.max(np.where(mask)[0]) #[int(pet_img.shape[0]*0.3), int(pet_img.shape[0]*0.7)]
        #     if z_min_mask < z_min:
        #         z_min = z_min_mask
        #     if z_max_mask > z_max:
        #         z_max = z_max_mask  

        #     y_min_mask, y_max_mask = np.min(np.where(mask)[1]),  np.max(np.where(mask)[1]) #[int(pet_img.shape[0]*0.3), int(pet_img.shape[0]*0.7)]
        #     if y_min_mask < y_min:
        #         y_min = y_min_mask
        #     if y_max_mask > y_max:
        #         y_max = y_max_mask 

        #     x_min_mask, x_max_mask = np.min(np.where(mask)[2]),  np.max(np.where(mask)[2]) #[int(pet_img.shape[0]*0.3), int(pet_img.shape[0]*0.7)]
        #     if x_min_mask < x_min:
        #         x_min = x_min_mask
        #     if x_max_mask > x_max:
        #         x_max = x_max_mask

            

        #     if (y_max - y_min) % 2 == 1: 
        #         y_max += 1
        #     if (x_max - x_min) % 2 == 1:
        #         x_max += 1
        #     delta = max(y_max - y_min, x_max-x_min) + 10
        #     if (delta) % 2 == 1:
        #         delta -= 1
        #     if (z_max - z_min ) % 2 == 1:
        #         z_min -= 1    

        #     pad_y = int((delta - (y_max-y_min)) / 2)+20
        #     pad_x = int((delta - (x_max-x_min)) / 2)+20
            
        #     mask_dict[structure].mask_deconv = mask[z_min-4:z_max+4,y_min-pad_y:y_max+pad_y, x_min-pad_x:x_max+pad_x]
        #     break
        # #if structure not found, skip to next patient
        # if z_min == 1000:
        #     print(f"Structure not found.. skipping patient {patient_num}")
        #     with open(os.path.join(data_folder, str("sivim" + patient_num + "_" + scan_type + "_MR_mask_dict")), "wb") as fp:
        #         pickle.dump(mask_dict, fp)

        #     with open(img_series_path, "wb") as fp:
        #         pickle.dump(img_dict_full, fp)
        # else:

        
        # # plot_3d_image(img[0,:,:,:])
        #     img = img[:,z_min-4:z_max+4,y_min-pad_y:y_max+pad_y, x_min-pad_x:x_max+pad_x]
        #     img_dict.deconv_coords_cord = coords_array[:,z_min-4:z_max+4,y_min-pad_y:y_max+pad_y, x_min-pad_x:x_max+pad_x]


        #     prediction_blurred, prediction_deblurred, kernel, img_max, noise = blind_deconvolution(img, patient_num, load_models=False, side="cord", scan_type=scan_type)
        #     img_dict.deconv_array_cord = prediction_deblurred
        #     img_dict.deconv_array_blurred_cord = prediction_blurred
        #     img_dict.img_cord = img
        #     img_dict.kernel_cord = kernel
        #     img_dict.max_cord = img_max
        #     img_dict.b_values = b_vals
        #     img_dict.noise_cord = noise






        

    
        with open(os.path.join(data_folder, str("sivim" + patient_num + "_" + scan_type + "_MR_mask_dict")), "wb") as fp:
            pickle.dump(mask_dict, fp)

        with open(img_series_path, "wb") as fp:
            pickle.dump(img_dict_full, fp)
    return

def prepare_diff_images(img_dict, mask_dict, patient_num, try_load=False):
    if try_load:
        try:
            with open(os.path.join(os.getcwd(), "cache", f"sivim_{patient_num}_training_images"), "rb") as fp:
                print("Loaded prepared training images from cache.")
                return pickle.load(fp)
        except:
            print("Failed to load images from cache. Calculating from loaded image dictionaries..")    

    #reshape the array so that pixel spacing matches slice thickness. 
    img_arrays = img_dict.image_arrays
    coords = img_dict.coords_array_img
    voxel_spacing = img_dict.pixel_spacing
    voxel_spacing.append(float(img_dict.slice_thickness))
    origin = (float(coords[0,0,0,0]), float(coords[1,0,0,0]), float(coords[2,0,0,0]))
    new_voxel_spacing = [voxel_spacing[0], voxel_spacing[1], voxel_spacing[2]]#[float(img_dict.slice_thickness), float(img_dict.slice_thickness), float(img_dict.slice_thickness)]

    coords_array_ = img_dict.coords_array_img
    #now want to stack the images by b value, and each layer will be a separate channel during training
    img_stack = []
    b_vals = []
    for b in sorted(list(img_arrays.keys())):
        img = img_arrays[b]
        #plot_3d_image(img[:,:,:])
        img = sitk.GetImageFromArray(img)
        img.SetOrigin(origin)
        img.SetSpacing(voxel_spacing)

        #reshape
        new_size = [int(sz * osp / nsp) for sz, osp, nsp in zip(img.GetSize(), voxel_spacing, new_voxel_spacing)]

        resampled_img = sitk.Resample(img, new_size, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(),
                                new_voxel_spacing, img.GetDirection(), 0.0, img.GetPixelID())
        img = sitk.GetArrayFromImage(resampled_img)
        #plot_3d_image(coords_array_[2,:,:,:])
        
        
        img_stack.append(img)
        b_vals.append(b)    
    img = np.stack(img_stack, axis=0)

    coords_stack = []
    for i in range(3):
        coords_array = sitk.GetImageFromArray(coords_array_[i,:,:,:])
        coords_array.SetOrigin(origin)
        coords_array.SetSpacing(voxel_spacing)
        coords_array = sitk.Resample(coords_array, new_size, sitk.Transform(), sitk.sitkLinear, coords_array.GetOrigin(),
                                    new_voxel_spacing, coords_array.GetDirection(), 0.0, coords_array.GetPixelID())
        coords_array = sitk.GetArrayFromImage(coords_array)
        coords_stack.append(coords_array)
    coords_array = np.stack(coords_stack, axis=0)
   # plot_3d_image(coords_array[2,:,:,:])

    for structure in mask_dict:
        mask = mask_dict[structure].whole_roi_masks
        mask = sitk.GetImageFromArray(mask.astype(int))
        mask.SetOrigin(origin)
        mask.SetSpacing(voxel_spacing)
        mask_resampled = sitk.Resample(mask, new_size, sitk.Transform(), sitk.sitkLinear, mask.GetOrigin(),
                                new_voxel_spacing, mask.GetDirection(), 0.0, mask.GetPixelID())
        mask_resampled = sitk.GetArrayFromImage(mask_resampled)
        #plot_3d_image(mask_resampled)
        mask_dict[structure].whole_roi_masks_resampled = mask_resampled

    with open(os.path.join(os.getcwd(), "cache", f"sivim_{patient_num}_training_images"), "wb") as fp:
        pickle.dump([img,mask_dict, b_vals],fp)
    return img, mask_dict, coords_array, b_vals
    


def blind_deconvolution(img, patient, load_models=False, load_kernel=False, save_dir=None, plot=False, side="left", scan_type="pre"):
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "models_ivim")
    #first normalize the images 
    img_thres = deepcopy(img)

    img_thres[img_thres < 50] = np.nan    #dont include outside body in stats
    img_std = np.nanstd(img_thres[0,:,:,:])
    img_mean = np.nanmean(img_thres[0,:,:,:])
    img_np = ((img - img_mean) / img_std)
    norm_min = np.amin(img_np)
    img_np = img_np - norm_min
    noise_vals = img_np[img < 40]
    noise_mean = np.nanmean(noise_vals)
    noise_std = np.nanstd(noise_vals)
    max_noise = noise_mean + 3*noise_std
    #plot_3d_image(pet_img)
    print(max_noise)
    # pet_img /= pet_img_sum
    # ct_img /= ct_img_sum
    if plot == True:
        plot_3d_image(img_np[0,:,:,:])
     
    img = np_to_torch(img_np)
    print(patient)
    print(save_dir)

    lr = 0.001
    lr_kernel = 8e-4
    num_iters=5001

    #put on gpu if possible
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU instead.")

    Gx = model_deconv_ivim.Gx(num_input_channels=16, 
                            num_output_channels=16, 
                            upsample_mode='trilinear', 
                            need_sigmoid=False, need_bias=True,act_fun='LeakyReLU').float()
    Gk = model_deconv.fcn(3179, 3179).float().to(device)
    np.random.seed(0)
    Gx_in = torch.from_numpy(np.random.uniform(0,1,(1,16,img.shape[2], img.shape[3], img.shape[4]))).float()
    #Gx_in = torch.from_numpy(np.zeros((1,1,pet_img.shape[2], pet_img.shape[3], pet_img.shape[4]))).float()
    #Gx_in = pet_img.unsqueeze(0)
    Gx.to(device)
    Gx_in = Gx_in.to(device)   
    img = img.to(device)
    


    print("Beginning blind deconvolution optimization...") 
    #plot_3d_image(out_k.squeeze().detach().cpu().numpy())  
    #do the first down-sampled loop
    ####################################################################################################
    small_Gk_in = torch.abs(torch.from_numpy(generate_target_kernel(kernel_size=5)).view(-1)).float().to(device)  
    small_img = zoom(Gx_in.detach().squeeze().cpu().numpy(), zoom=(1, 0.5, 0.5, 0.5), order=1)
    small_img = np_to_torch(small_img).to(device)
    Gk_small = model_deconv.fcn(5**3, 5**3).float().to(device)
    target = np_to_torch(zoom(img_np, zoom=(1,0.5,0.5,0.5), order=1)).unsqueeze(0).to(device)
    optimizer_small = torch.optim.Adam([{'params': Gx.parameters()}, {'params': Gk_small.parameters(), 'lr': lr_kernel}], lr=lr)
    loss_func_small = loss_functions_ivim.Deconv_Loss_Small(device=device)
    #we want the kernel model to start off by PREDICTING a gaussian kernel. so we need to have this kernel model pretrained
    kernel_optimizer = torch.optim.Adam([{'params': Gk_small.parameters(), 'lr': lr_kernel}])
    kernel_loss = loss_functions_ivim.Kernel_Loss()
    y= deepcopy(small_Gk_in.view(1,1, 5,5,5).float())
    #plot_3d_image(target.detach().squeeze().cpu().numpy())
    #plot_3d_image(generate_target_kernel(kernel_size=7))
    #plot_3d_image(y.cpu().detach().squeeze().numpy())
    Gx_optimizer = torch.optim.Adam([{'params': Gx.parameters(), 'lr': lr}])
    Gx_loss = loss_functions_ivim.Kernel_Loss()
    pre_loss = []
    #pre train the Gx model to predict last one too 
    for pre_train_step in range(300):
        Gx_optimizer.zero_grad()
        out_x = Gx(small_img).float()
        loss = Gx_loss(out_x, target)
        loss.backward()
        Gx_optimizer.step()
        # if pre_train_step % 10 == 0:
        #     print(f"step: {pre_train_step} - Gx loss: {round(loss.item(),9)}")

    # #print("predicted Kernel 2 - pre-training")    
    #plot_3d_image(out_x.squeeze().detach().cpu().numpy())

    for pre_train_step in range(100):
        kernel_optimizer.zero_grad()
        out_k = Gk_small(small_Gk_in).view(1,1, 5,5,5).float()
        loss = kernel_loss(out_k, y)
        loss.backward()
        kernel_optimizer.step()

    start_time = time.time()    
    for small_step in range(1000):
        optimizer_small.zero_grad()
        out_x = Gx(small_img).float()
        out_k = Gk_small(small_Gk_in).view(1,1, 5,5,5).float()#.repeat(16,16,1,1,1)

        #sum = torch.sum(out_k)
        #print(sum)
        #prediction = nn.functional.conv3d(out_x, out_k, padding=(3,3,3), bias=None).float()
        prediction_interp = out_x
        prediction = torch.zeros_like(prediction_interp)
        for j in range(16):
            prediction[0,j,:,:,:] = nn.functional.conv3d(prediction_interp[0,j,:,:,:].unsqueeze(0).unsqueeze(0), out_k, padding=(2,2,2), bias=None).float()
        #prediction = nn.functional.conv3d(prediction_interp, out_k, padding=(3,3,3), bias=None).float()
        loss = loss_func_small(prediction, target, out_k, prediction_interp, small_step) #, fidelity_term, norm_term, tv_loss, tv_loss_mask, loss_kernel
        loss.backward()
        pre_loss.append(loss.item())
        optimizer_small.step()

        if small_step % 100 == 0:
            print(f"""Finished step {small_step}, loss = {round(loss.item(),10)}""")#, fidelity = {round(fidelity_term.item(),6)}, nor
            print(f"lr: {optimizer_small.param_groups[0]['lr']}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    mins = int(elapsed_time / 60)
    secs = int(elapsed_time % 60)
    print(f"Elapsed time: {mins} minutes and {secs} seconds")
    #do the second down-sampled loop
    #########################################################################################################
    small_Gk_in = out_k[0,0,:,:,:].cpu().detach().numpy()
    print("predicted Kernel 1")  
    if plot ==True: 
        plot_3d_image(small_Gk_in)  
    small_Gk_in = zoom(small_Gk_in, zoom=[7/5,7/5,7/5], order=1) / np.sum(small_Gk_in)
    small_Gk_in = torch.abs(np_to_torch(small_Gk_in)).view(-1).float().to(device)    
    
    if plot ==True: 
        print("Kernel 1 zoomed")   
        plot_3d_image(torch.reshape(small_Gk_in,(7,7,7)).detach().cpu().numpy())
    
    small_img = out_x.squeeze().detach().cpu().numpy()
    if plot ==True: 
        print("image 1")   
        plot_3d_image(small_img[0,:,:,:])
        plot_3d_image(small_img[3,:,:,:])
        plot_3d_image(small_img[10,:,:,:])
    small_img = zoom(small_img, zoom=(1,2**0.5,2**0.5,2**0.5), order=1)
    if plot ==True: 
        print("image 1 interp")   
        plot_3d_image(small_img[0,:,:,:])

    small_img = np_to_torch(small_img).to(device)
    Gk_small = model_deconv.fcn(7**3, 7**3).float().to(device)
    target = np_to_torch(zoom(img_np, zoom=(1,2**-0.5,2**-0.5,2**-0.5), order=1)).to(device)
    optimizer_small = torch.optim.Adam([{'params': Gx.parameters()}, {'params': Gk_small.parameters(), 'lr': lr_kernel}], lr=0.001)
    loss_func_small = loss_functions_ivim.Deconv_Loss_Small(device=device)
    #we want the kernel model to start off by PREDICTING a gaussian kernel. so we need to have this kernel model pretrained
    kernel_optimizer = torch.optim.Adam([{'params': Gk_small.parameters(), 'lr': lr_kernel}])
    kernel_loss = loss_functions_ivim.Kernel_Loss()
    y = deepcopy(small_Gk_in.view(-1,1, 7,7,7).float())

    for pre_train_step in range(100):
        kernel_optimizer.zero_grad()
        out_k = Gk_small(small_Gk_in.cuda()).view(-1,1, 7,7,7).float()
        loss = kernel_loss(out_k, y)
        loss.backward()
        kernel_optimizer.step()
        if pre_train_step % 100 == 0:
            print(f"step: {pre_train_step} - Kernel loss: {round(loss.item(),10)}")

    Gx_optimizer = torch.optim.Adam([{'params': Gx.parameters(), 'lr': 0.002}])
    Gx_loss = nn.MSELoss()
    print("predicted Kernel 2 - pre-training")    
    #plot_3d_image(out_k.squeeze().detach().cpu().numpy())
    #pre train the Gx model to predict last one too 
    for pre_train_step in range(300):
        Gx_optimizer.zero_grad()
        out_x = Gx(small_img).float()
        loss = Gx_loss(out_x, small_img)
        loss.backward()
        Gx_optimizer.step()
        # if pre_train_step % 100 == 0:
        #     print(f"step: {pre_train_step} - Gx loss: {round(loss.item(),9)}")
    #print("predicted Image 2 - pre-training")    
    #plot_3d_image(out_x.squeeze().detach().cpu().numpy())

    for small_step_2 in range(1000):
        optimizer_small.zero_grad()
        out_x = Gx(small_img).float()
        out_k = Gk_small(small_Gk_in).view(-1,1, 7,7,7).float()#.repeat(16,16,1,1,1)

        #sum = torch.sum(out_k)
        #print(sum)
        #prediction = nn.functional.conv3d(out_x, out_k, padding=(5,5,5), bias=None).float()
        #prediction = nn.functional.conv3d(out_x, out_k, padding=(5,5,5), bias=None).float()
        prediction_interp = out_x
        prediction = torch.zeros_like(prediction_interp)
        for j in range(16):
            prediction[0,j,:,:,:] = nn.functional.conv3d(prediction_interp[0,j,:,:,:].unsqueeze(0).unsqueeze(0), out_k, padding=(3,3,3), bias=None).float()
        #prediction = nn.functional.conv3d(prediction_interp, out_k, padding=(5,5,5), bias=None).float()
        loss = loss_func_small(prediction, target, out_k, prediction_interp, small_step) #, fidelity_term, norm_term, tv_loss, tv_loss_mask, loss_kernel
        # prediction_interp = F.interpolate(out_x, scale_factor=0.5, mode='trilinear')
        # prediction = nn.functional.conv3d(prediction_interp, out_k, padding=(5,5,5), bias=None).float()

        # loss = loss_func_small(prediction, target, out_k, prediction_interp, small_step_2) #, fidelity_term, norm_term, tv_loss, tv_loss_mask, loss_kernel
        loss.backward()
        pre_loss.append(loss.item())
        optimizer_small.step()
        if small_step_2 % 100 == 0:
            print(f"""Finished step {small_step_2}, loss = {round(loss.item(),8)}""")#, fidelity = {round(fidelity_term.item(),6)}, nor
            print(f"lr: {optimizer_small.param_groups[0]['lr']}")
    
    with open(os.path.join(save_dir, f"{patient}_pre_loss_{side}.txt"), "wb") as fp:
        pickle.dump(pre_loss, fp)
    
    #now upsample the kernel and image back to original size and train models to start with that.
    #############################################################################################################################
    Gk_in = out_k[0,0,:,:,:].cpu().detach().numpy()
    if plot ==True: 
        print("Final starting kernel")
        plot_3d_image(Gk_in[:,:,:])
    Gk_in = zoom(Gk_in, zoom=[11/7,11/7,11/7], order=1)
    Gk_in = torch.abs(np_to_torch(Gk_in).view(-1)).float().to(device) 
    if plot ==True: 
        print("Final starting kernel - zoomed")
        plot_3d_image(torch.reshape(Gk_in,(11,11,11)).detach().cpu().numpy())
    
    if plot ==True: 
        print("image 1 interp")   
        plot_3d_image(out_x.detach().squeeze().cpu().numpy()[0,:,:,:])

    Gx_in = out_x.detach().squeeze().cpu().numpy()
    
    if plot ==True: 
        print("Final starting image")
        plot_3d_image(Gx_in[0,:,:,:])
    Gx_in = zoom(Gx_in, zoom=[1, img.shape[2]/out_x.shape[2],img.shape[3]/out_x.shape[3],img.shape[4]/out_x.shape[4]], order=1)
    # Gx_random_in = np.random.rand(*Gx_in.shape).astype(np.float32)*0.02
    # Gx_in = np.concatenate((Gx_in, Gx_random_in), axis=0)
    #plot_3d_image(Gx_in[16,:,:,:])
    Gx_in = np_to_torch(Gx_in).to(device)
    
    #delete old model and clear cache
    torch.cuda.empty_cache()
    del Gk_small
    Gk = model_deconv.fcn(11**3, 11**3).float().to(device)

    #we want the kernel model to start off by PREDICTING a gaussian kernel. so we need to have this kernel model pretrained
    kernel_optimizer = torch.optim.Adam([{'params': Gk.parameters(), 'lr': lr_kernel}])
    kernel_loss = loss_functions_ivim.Kernel_Loss()
    y= deepcopy(Gk_in.view(-1,1, 11,11,11).float())
    print("final kernel y")

    Gx = model_deconv_ivim.Gx(num_input_channels=16, 
                            num_output_channels=16, 
                            upsample_mode='trilinear', 
                            need_sigmoid=False, need_bias=True,act_fun='LeakyReLU').float().to(device)
    

    #plot_3d_image(y.squeeze().detach().cpu().numpy())
    for pre_train_step in range(100):
        kernel_optimizer.zero_grad()
        out_k = Gk(Gk_in.cuda()).view(-1,1, 11,11,11).float()
        loss = kernel_loss(out_k, y)
        loss.backward()
        kernel_optimizer.step()
        # if pre_train_step % 10 ==0 :
        #     print(f"Step: {pre_train_step} | Kernel loss: {round(loss.item(),9)}")


    Gx_optimizer = torch.optim.Adam([{'params': Gx.parameters(), 'lr': lr}])
    Gx_loss = loss_functions_ivim.Kernel_Loss()

    num_voxels = Gx_in.shape[1]*Gx_in.shape[2]* Gx_in.shape[3] * Gx_in.shape[4]
    Gnoise = model_deconv_ivim.noise_net(num_voxels).to(device)
    Gnoise_in = torch.from_numpy(np.random.uniform(0,1,(1,num_voxels))).float().to(device) * max_noise
    #now want a noise term of same size as the input image. 
    #pre train the Gx model to predict last one too 
    for pre_train_step in range(300):
        Gx_optimizer.zero_grad()
        out_x = Gx(Gx_in)
        loss = Gx_loss(out_x, Gx_in)
        loss.backward()
        Gx_optimizer.step()
        # if pre_train_step % 10 == 0:
        #     print(f"step: {pre_train_step} - Gx loss: {round(loss.item(),9)}")

    # if plot ==True: 
    #     print("predicted Kernel final - pre-training")    
    #     plot_3d_image(out_k.squeeze().detach().cpu().numpy())
    #     print("predicted Image final - pre-training")    
    #     plot_3d_image(out_x.squeeze().detach().cpu().numpy())
    # if plot ==True: 
    #     print("pet image - pre-training")    ma

    #     plot_3d_image(pet_img.squeeze().detach().cpu().numpy())
    
    lr = 0.001  
    
    torch.cuda.empty_cache()
    optimizer = torch.optim.Adam([{'params': Gx.parameters()},{'params': Gnoise.parameters()}, {'params': Gk.parameters(), 'lr': lr_kernel}], lr)
    loss_func = loss_functions_ivim.Deconv_Loss(device=device, max_noise=max_noise)
    #scheduler = MultiStepLR(optimizer, milestones=[1000,2000,3000,4000, 5000,6000], gamma=0.5)
    # scheduler = ReduceLROnPlateau(
    #     optimizer,          # The optimizer whose learning rate will be reduced
    #     mode='min',         # Monitor the 'min' value of the metric (in this case, the loss)
    #     factor=0.7,         # Factor by which the learning rate will be reduced (new_lr = lr * factor)
    #     patience=100,         # Number of epochs with no improvement after which the learning rate will be reduced
    #     min_lr=1e-5         # Lower bound on the learning rate
    # )
    scheduler = MultiStepLR(optimizer, milestones=[2000, 4000], gamma=0.5)
    loss_history = []
    initial_step = 0

    #noise_term = torch.from_numpy(np.random.uniform(0, 0.02, size=Gx_in.shape[1:])).unsqueeze(0)
    noise_init = Gnoise(Gnoise_in).view(img.shape).squeeze().detach().cpu().numpy()
    
    start_time = time.time()
    for step in range(initial_step, num_iters):
        optimizer.zero_grad()
        out_x = Gx(Gx_in).float()
        out_k = Gk(Gk_in).view(-1,1, 11,11,11).float()#.repeat(16,16,1,1,1)
        out_noise = Gnoise(Gnoise_in).view(out_x.shape)
        #sum = torch.sum(out_k)
        #print(sum)
        #prediction = nn.functional.conv3d(out_x, out_k, padding=(7,7,7), bias=None).float()
        #prediction = nn.functional.conv3d(out_x, out_k, padding=(7,7,7), bias=None).float()

        prediction = torch.zeros_like(out_noise)

        for j in range(16):
            prediction[0,j,:,:,:] = out_noise[0,j,:,:,:] + nn.functional.conv3d(out_x[0,j,:,:,:].unsqueeze(0).unsqueeze(0), out_k, padding=(5,5,5), bias=None).float()
        
        #prediction = nn.functional.conv3d(prediction_interp, out_k, padding=(7,7,7), bias=None).float()
        loss = loss_func(prediction, img, out_k, out_x,  out_noise, step) #, fidelity_term, norm_term, tv_loss, tv_loss_mask, loss_kernel
        # prediction_interp = F.interpolate(out_x, scale_factor=0.5, mode='trilinear')
        # prediction = nn.functional.conv3d(prediction_interp, out_k, padding=(7,7,7), bias=None).float()
        # loss = loss_func(prediction, pet_img, out_k, prediction_interp, step) #, fidelity_term, norm_term, tv_loss, tv_loss_mask, loss_kernel
        loss_history.append(loss.item())#, fidelity_term.item(), norm_term.item(), tv_loss.item(),tv_loss_mask.item(), loss_kernel.item()])
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            print(f"""Finished step {step}, loss = {round(loss.item(),8)}""")#, fidelity = {round(fidelity_term.item(),6)}, norm = {round(norm_term.item(),6)}, tv loss = {round(tv_loss.item(),6)}, tv mask loss = {round(tv_loss_mask.item(),6)}, kernel tv loss = {(round(loss_kernel.item(),6))}""")
            print(f"lr: {optimizer.param_groups[0]['lr']}")
            if step > 0 and step % 1000 == 0: 
                #print the time taken 
                end_time = time.time()
                elapsed_time = end_time - start_time
                mins = int(elapsed_time / 60)
                secs = int(elapsed_time % 60)
                start_time = time.time()

                
                print(f"Elapsed time: {mins} minutes and {secs} seconds")
                
                if plot == True: 
                
                    prediction = prediction.squeeze().detach().cpu().numpy()
                    img_view = img.squeeze().detach().cpu().numpy()
                    out_k = out_k[0,0,:,:,:].detach().cpu().numpy()
                    out_x = out_x.squeeze().detach().cpu().numpy()
                    out_noise = out_noise.squeeze().detach().cpu().numpy()

                    
                    # shift, error, diff = phase_cross_correlation(img_view, out_x[:,:,:,:])
                    # #use middle slice of both images .
                    # out_x = shifter(out_x, [val for val in shift])
                    # #plot_3d_image(out_k)
                    # out_k = shifter(out_k, [val for val in shift[1:]])

                    plot_3d_image(out_k)
                    for idx in [0,5,12]:
                        plot_3d_image(img_view[idx,:,:,:])
                        plot_3d_image(out_x[idx,:,:,:])
                        plot_3d_image(out_noise[idx,:,:,:])
                

        

    prediction = prediction.squeeze().detach().cpu().numpy()
    img_view = img.squeeze().detach().cpu().numpy()
    out_k = out_k[0,0,:,:,:].squeeze().detach().cpu().numpy()
    out_x = out_x.squeeze().detach().cpu().numpy()
    out_noise = out_noise.squeeze().detach().cpu().numpy()
    # shift, error, diff = phase_cross_correlation(img_view, out_x[:,:,:,:])
    # #use middle slice of both images .
    # out_x = shifter(out_x, [val for val in shift], order=1)
    # #plot_3d_image(out_k)
    # out_k = shifter(out_k, [val for val in shift[1:]], order=1)

    with open(os.path.join(save_dir,  f"{patient}_model_stuff_{side}_{scan_type}.txt"), "wb") as fp:
        pickle.dump([((prediction+ norm_min) * img_std) + img_mean, ((out_x+ norm_min) * img_std) + img_mean, out_k, ((img+ norm_min) * img_std) + img_mean, ((out_noise+ norm_min) * img_std) + img_mean, loss_history, step], fp)
    torch.save({'model_state_dict': Gx.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, os.path.join(save_dir, f"{patient}_Gx_with_optimizer_{side}_{scan_type}.pt"))  
    torch.save({'model_state_dict': Gk.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, os.path.join(save_dir, f"{patient}_Gk_with_optimizer_{side}_{scan_type}.pt"))    
    return ((prediction+ norm_min) * img_std) + img_mean, ((out_x+ norm_min) * img_std) + img_mean, out_k, ((img_view+ norm_min) * img_std) + img_mean, ((out_noise+ norm_min) * img_std) + img_mean
    
if __name__ == "__main__":
    main(patient_list=["10", "11", "12", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23'], scan_type="pre")
    main(patient_list=["10", "11", "12", "13", "16"], scan_type="post")
    
    # main(patient_list=["10","11","12","13"], scan_type="pre")
    
    

