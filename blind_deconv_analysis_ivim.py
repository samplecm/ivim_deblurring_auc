import os 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import pickle
import SimpleITK as sitk
import torch
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing, binary_fill_holes, label, zoom 
from scipy.ndimage import shift as shifter
from scipy.signal import correlate
from utils import *
import loss_functions
import model_deconv
import torch.nn.functional as F
import radiomics
import six 
import cv2

from copy import deepcopy
from utils import *
import torch.nn.functional as F
from blind_deconv_ivim import blind_deconvolution
from skimage.registration import phase_cross_correlation
import matplotlib.ticker as ticker
import get_contour_masks
from piq import brisque, CLIPIQA, TVLoss
import mplcursors
from copy import deepcopy
from scipy.stats import ttest_rel, spearmanr
import ast


data_folder = os.path.join(os.getcwd(), "data")
def main():
    #plot_diffusion_img_with_signal_curve(scan_type="pre")
   # verify_noise_images()
    #register_predicted_pseudokernels()
    validate_pseudo_kernels()
    #register_predicted_images()
    #re_train_with_fake_kernels()
    #assess_signal_decay_curves()
    #compare_predicted_kernels()
    #get_image_quality_score(img_type="orig")
    # get_image_quality_score(img_type="adc")
    # get_image_quality_score(img_type="auc")
    # get_image_quality_score(img_type="auc_l")
    # get_image_quality_score(img_type="auc_m")
    # get_image_quality_score(img_type="auc_h")
    # for i in range(3):
    #     get_image_quality_score(img_type="biexp", ind=i)
    # for i in range(5):
    #     get_image_quality_score(img_type="triexp", ind=i)
   # plot_auc_vs_params()

    #plot_max_curvature_location_vs_params()
    #plot_diffusion_img_with_signal_curve()
    #plot_original_and_deblurred()

    #create_uptake_line_plot()
    #plot_original_and_deblurred()
    #plot_all_parameters(deblurred=True)
    #plot_predicted_kernels()
    #get_performance_stats()
    #eigen_kernels(include_spacing_type="both")
    #get_patient_voxel_stats()
    #get_gland_stats()
    #plot_loss_history()
    #get_image_quality_score()

def assess_signal_decay_curves():
    patient_nums = ["10", "11", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']
    scan_type = "pre"

    ranks = np.linspace(1,16,16)
    corrs = []
    corrs_deblur = []
    for patient_num in patient_nums[0:]:
        print(f"on patient {patient_num}")
        img_series_path = os.path.join(os.getcwd(), "data_mri")
        img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + "pre" + "diffusion_imgs"))
        with open(img_series_path, "rb") as fp:
            img_dict_full = pickle.load(fp)
        img_dict = img_dict_full["diffusion"]

        b_values = list(img_dict.image_arrays.keys())
        b_values = np.array(b_values)
        for roi in ["r_par", "l_par"]:
            try:

                out_x = getattr(img_dict, f"deconv_array_{roi}")
                img = getattr(img_dict, f"img_{roi}")


            except:
                continue
            for s in range(int(img.shape[1]/2),int(img.shape[1]/2)+2):
                print(f"Slice {s}")
                for y in range(img.shape[2]):
                    for x in range(img.shape[3]):
                        if img[0,s,y,x] < 50:
                            continue
                        voxels = img[:,s,y,x]
                        voxels_deblur = out_x[:,s,y,x]

                        voxels_corr, _ = spearmanr(b_values, voxels)
                        voxels_deblur_corr, _ = spearmanr(b_values, voxels_deblur)

                        corrs.append(voxels_corr)
                        corrs_deblur.append(voxels_deblur_corr)
    mean_corr = np.mean(corrs)
    mean_corr_deblur = np.mean(corrs_deblur)
    std_corr = np.std(corrs)
    std_corr_deblur = np.std(corrs_deblur)

    t, p = ttest_rel(corrs, corrs_deblur)
    print(mean_corr, std_corr, "|", mean_corr_deblur, std_corr_deblur, "|", t, p )

def re_train_with_fake_kernels(kernel_type="gaussian",stretch_factors=[1,3,1], load_kernels=False):
    #here we take the deblurred images that have been previously predicted, and apply fake blur kernels, and then re-run the algorithm, to see how close to the fake kernel, the kernel prediction is
    patient_nums = ["10", "11", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']
    scan_type = "pre"
    
    #plot_3d_image(blur_kernel)
    kernels = {} #store predicted pseudokernels for all patients 
    noises = {}
    for patient_num in patient_nums[0:]:
        kernels[patient_num] = {}
        noises[patient_num] = {}
        for stretch_factors in [[1,1,1], [1,3,1], [1,1,3], [3,1,1]]:

            img_series_path = os.path.join(os.getcwd(), "data_mri")
            img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + "pre" + "diffusion_imgs"))
            with open(img_series_path, "rb") as fp:
                img_dict_full = pickle.load(fp)
            img_dict = img_dict_full["diffusion"]

            b_values = list(img_dict.image_arrays.keys())
            b_values = np.array(b_values)
            for roi in ["r_par"]:
                try:

                    out_x = getattr(img_dict, f"deconv_array_{roi}")
                    noise = getattr(img_dict, f"noise_{roi}")

                except:
                    continue

                blur_kernel = generate_target_kernel(kernel_size=11, kernel_type=kernel_type, peak_width=1.5, stretch_factors=stretch_factors)

                #first step is to define the kernel that we will convolve with 
                
                #downsample out_x to the original pet size (2x smaller)
                #now convolve with the original kernel
                #plot_3d_image(out_x[0,...])
                out_x = np_to_torch(out_x)
                blur_kernel = np_to_torch(blur_kernel).unsqueeze(0).float()
                #out_k = np_to_torch(out_k).unsqueeze(0)
                noise= np_to_torch(noise).float()
                
                blur_img = torch.zeros_like(noise)
                for j in range(16):
                    blur_img[0,j,:,:,:] = nn.functional.conv3d(out_x[0,j,:,:,:].unsqueeze(0).unsqueeze(0), blur_kernel, padding=(5,5,5), bias=None).float()

                blur_img = blur_img.squeeze().detach().cpu().numpy()
                #plot_3d_image(blur_img[0,:,:,:])


                #now rerun the deblurring program with this new 
                save_dir = os.path.join(os.getcwd(), "models_pseudokernel_ivim")
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                _, new_x, new_k, _, new_noise = blind_deconvolution(blur_img, patient_num, save_dir=save_dir, side="right", scan_type="pre")
                kernels[patient_num][str(stretch_factors)] = [new_k, new_x, new_noise, out_x]

                with open(os.path.join(save_dir, f"all_pseudokernels.txt"), "wb") as fp:
                    pickle.dump(kernels, fp)

def register_predicted_pseudokernels():

    with open(os.path.join(os.getcwd(), "models_pseudokernel_ivim", f"all_pseudokernels.txt"), "rb") as fp:
        data = pickle.load(fp)
    new_data ={}
    
    for patient in data: 

        new_data[patient] = {}
        for stretch_factor in  data[patient]:
            fake_kernel = generate_target_kernel(kernel_size=11 ,kernel_type='gaussian', peak_width=0.75, stretch_factors=ast.literal_eval(stretch_factor))
            new_k, new_x, new_noise, old_x = data[patient][stretch_factor]
            old_x = old_x.squeeze().detach().cpu().numpy()
            from skimage.registration import phase_cross_correlation
            #shift, error, diff = phase_cross_correlation(old_x, new_x)
            # #use middle slice of both images .
            # new_x = shifter(new_x, [val for val in shift])
            # new_noise = shifter(new_noise, [val for val in shift])
            # #plot_3d_image(new_k)
            # new_k = shifter(new_k, [val for val in shift[1:]])

            shift, error, diff = phase_cross_correlation(fake_kernel, new_k)
            new_k = shifter(new_k, [val for val in shift])


            new_data[patient][stretch_factor] = [new_k, new_x, new_noise, old_x]
    with open(os.path.join(os.getcwd(),"models_pseudokernel_ivim",  f"registered_pseudokernel_predictions.txt"), "wb") as fp:
        pickle.dump(new_data, fp)
    return

def validate_pseudo_kernels():
    
    fig, ax = plt.subplots(4,6, figsize=(20,20))
    
    cmap="inferno"

    # plot_3d_image(img, view='cor')
    with open(os.path.join(os.getcwd(), "models_pseudokernel_ivim", f"registered_pseudokernel_predictions.txt"), "rb") as fp:
        data = pickle.load(fp)
    # for key in data['02']:    
    #     kernel = data['02'][key][0]
    #     kernel[kernel < 0.001] = np.nan
    #     plot_3d_kernel(kernel)
    #     plot_3d_image(data['02'][key][0], view='sag')
    avg_kernels = {}

    for s, stretch_factor in enumerate([[1,1,1], [1,3,1], [1,1,3], [3,1,1]]):
        avg_kernels[str(stretch_factor)] = {}
        combined_kernel_image = np.zeros([11,11,11])

        fake_kernel = generate_target_kernel(kernel_size=11 ,kernel_type='gaussian', peak_width=1.5, stretch_factors=stretch_factor)
        fake_axial_projection = np.round(np.sum(fake_kernel, axis=0),2)
        fake_sagittal_projection = np.round(np.sum(fake_kernel,axis=2),2)
        fake_coronal_projection = np.round(np.sum(fake_kernel, axis=1),2)

        fake_kernel = deepcopy(fake_kernel)

        inner_product_vals = []
        kernel_count = 0
        for k1, kern_1 in enumerate(list(data.keys())):
            try:
                kernel_1 = deepcopy(data[kern_1][str(stretch_factor)][0])
                kernel_count += 1
            except:
                continue    
            combined_kernel_image += kernel_1
    
            norm_1 = np.sqrt(np.sum(kernel_1 **2))
            norm_2 = np.sqrt(np.sum(fake_kernel**2))
            kernel_1 /= norm_1
            fake_kernel /= norm_2
            inner_product = np.sum(kernel_1 * fake_kernel)
            inner_product_vals.append(inner_product)

        combined_kernel_image /= kernel_count
        combined_kernel_image /= np.sqrt(np.sum(combined_kernel_image**2))
        print(np.sum(combined_kernel_image*fake_kernel))
        combined_kernel_image /= np.sum(combined_kernel_image)
        #combined_kernel_image[combined_kernel_image < 0.001] = np.nan
        #plot_3d_kernel(combined_kernel_image)
        axial_projection = np.round(np.sum(combined_kernel_image, axis=0),8)
        sagittal_projection = np.round(np.sum(combined_kernel_image,axis=2),8)
        coronal_projection = np.round(np.sum(combined_kernel_image, axis=1),8)

        

        im_0 = ax[s,0].imshow(fake_axial_projection, cmap=cmap, vmin=0, vmax=0.21)
        im_1 = ax[s,1].imshow(fake_coronal_projection, cmap=cmap, vmin=0, vmax=0.21)
        im_2 = ax[s,2].imshow(fake_sagittal_projection, cmap=cmap, vmin=0, vmax=0.21)

        im_3 = ax[s,3].imshow(axial_projection, cmap=cmap, vmin=0, vmax=0.72)
        im_4 = ax[s,4].imshow(coronal_projection, cmap=cmap, vmin=0, vmax=0.72)
        im_5 = ax[s,5].imshow(sagittal_projection, cmap=cmap, vmin=0, vmax=0.72)

        cbar_1 = fig.colorbar(im_0,ax=ax[s,0])
        cbar_2 = fig.colorbar(im_1,ax=ax[s,1])
        cbar_3 = fig.colorbar(im_2,ax=ax[s,2])
        cbar_4 = fig.colorbar(im_3,ax=ax[s,3])
        cbar_5 = fig.colorbar(im_4,ax=ax[s,4])
        cbar_6 = fig.colorbar(im_5,ax=ax[s,5])

        cbar_1.set_ticks([np.amin(fake_axial_projection), np.amax(fake_axial_projection)])
        cbar_2.set_ticks([np.amin(fake_coronal_projection), np.amax(fake_coronal_projection)])
        cbar_3.set_ticks([np.amin(fake_sagittal_projection),np.amax(fake_sagittal_projection)])
        cbar_4.set_ticks([np.round(np.amin(axial_projection),2)+0.01, np.round(np.amax(axial_projection),2)-0.01])
        cbar_5.set_ticks([np.round(np.amin(coronal_projection),2)+0.01, np.round(np.amax(coronal_projection),2)-0.01])
        cbar_6.set_ticks([np.round(np.amin(sagittal_projection),2)+0.01, np.round(np.amax(sagittal_projection),2)-0.01])
        
        mean = np.mean(inner_product_vals)
        print(mean)
    for a in ax.flat:
        
        a.tick_params(axis='both', which='both', length=0)
        a.set_xticks([])
        a.set_yticks([])    
    plt.tight_layout()
    plt.show(block=True)
    return

def plot_all_parameters(patient="15", roi="l_par", deblurred=False, smoothed=False, scan_type="pre", cmap="plasma"):
    #this creates a 2x2 subplot of the original and deblurred image, with a line across the body, and below a plot of the uptake vs distance across the line. can then compare max gradient, etc. 
    img_series_path = os.path.join(os.getcwd(), "data_mri")
    img_series_path = os.path.join(img_series_path, str("SIVIM" + patient + "pre" + "diffusion_imgs"))
    with open(img_series_path, "rb") as fp:
        img_dict_full = pickle.load(fp)
    img_dict = img_dict_full["diffusion"]
    if deblurred == True:
        img = getattr(img_dict, f"deconv_array_{roi}")
    else:
        img = getattr(img_dict, f"img_{roi}")
    auc_img = getattr(img_dict, f"auc_img_{roi}_{deblurred}_{smoothed}_{False}_{scan_type}")
    auc_l_img = getattr(img_dict, f"auc_l_img_{roi}_{deblurred}_{smoothed}_{False}_{scan_type}")
    auc_m_img = getattr(img_dict, f"auc_m_img_{roi}_{deblurred}_{smoothed}_{False}_{scan_type}")
    auc_h_img = getattr(img_dict, f"auc_h_img_{roi}_{deblurred}_{smoothed}_{False}_{scan_type}")
    auc_s0_img = getattr(img_dict, f"auc_img_{roi}_{deblurred}_{smoothed}_{True}_{scan_type}")    #version where uses exponential s0 prediction
    auc_s0_l_img = getattr(img_dict, f"auc_l_img_{roi}_{deblurred}_{smoothed}_{True}_{scan_type}")
    auc_s0_m_img = getattr(img_dict, f"auc_m_img_{roi}_{deblurred}_{smoothed}_{True}_{scan_type}")
    auc_s0_h_img = getattr(img_dict, f"auc_h_img_{roi}_{deblurred}_{smoothed}_{True}_{scan_type}")
    adc_img = getattr(img_dict, f"adc_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
    triexp_img = getattr(img_dict, f"triexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")
    biexp_img = getattr(img_dict, f"biexp_img_{roi}_{deblurred}_{smoothed}_{scan_type}")

    slice_idx = int(img.shape[1] / 2)


    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(img[0,slice_idx, :,:])
    ax.set_xticks([])
    ax.set_yticks([])
    #cbar3.ax.xaxis.set_major_formatter(formatter)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(adc_img[0,slice_idx, :,:])
    ax.set_xticks([])
    ax.set_yticks([])
    #cbar3.ax.xaxis.set_major_formatter(formatter)
    plt.show()
    
    
    #biexp imgs:
    fig, axs = plt.subplots(1, 3, figsize=(30, 30/3))
    labels = ["$D^*$", "D", "f"]
    ims = []
    for i in range(3):
        ax = axs[i]
        maxs = [0.03, np.amax(biexp_img[i,slice_idx, :,:]),np.amax(biexp_img[i,slice_idx, :,:])]
        ims.append(ax.imshow(biexp_img[i,slice_idx, :,:],vmax=maxs[i]))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(labels[i], fontsize=20)

    cax1 = fig.add_axes([0.15, 0.85, 0.2, 0.02])  # Adjust the coordinates and size as needed
    cax2 = fig.add_axes([0.42, 0.85, 0.2, 0.02])
    cax3 = fig.add_axes([0.69, 0.85, 0.2, 0.02])

    # Add colorbars for each subplot
    cbar1 = plt.colorbar(ims[0], cax=cax1, orientation="horizontal")
    cbar2 = plt.colorbar(ims[1], cax=cax2, orientation="horizontal")
    cbar3 = plt.colorbar(ims[2], cax=cax3, orientation="horizontal")
    from matplotlib.ticker import FormatStrFormatter
    formatter = FormatStrFormatter('%.4f')
    #cbar1.ax.xaxis.set_major_formatter(formatter)
    cbar2.ax.xaxis.set_major_formatter(formatter)
    #cbar3.ax.xaxis.set_major_formatter(formatter)
    plt.show()

    fig, ax = plt.subplots(1,4, figsize=(30,30/4))
    labels = ["$AUC$", "$AUC_L$", "$AUC_M$", "$AUC_H$"]
    imgs = [auc_s0_img, auc_s0_l_img, auc_s0_m_img, auc_s0_h_img]
    mins = [400,90,100,200]
    maxs=[900,120,250,600]
    ims = []
    for i in range(4):
        img = imgs[i]
        ims.append(ax[i].imshow(img[slice_idx, :,:], vmin=mins[i], vmax=maxs[i]))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(labels[i], fontsize=20)

        # Define the positions and sizes for the colorbars
    num_subplots = 4
    cax_width = 0.2
    spacing = (1.0 - (num_subplots * cax_width)) / (num_subplots + 1)
    cax_positions = []
    for i in range(num_subplots):
        left = spacing * (i + 1) + i * cax_width
        cax_positions.append([left, 0.85, cax_width, 0.02])


    # Create axes for colorbars and position them above the subplots
    for i in range(4):
        cax = fig.add_axes(cax_positions[i])
        cbar = plt.colorbar(ims[i], cax=cax, orientation="horizontal")
        
    plt.show()
    

def verify_noise_images():
    #this function checks the distribution in noise images, to see if its rician or not.
    patient_nums = ["10", "11", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']
    scan_type = "pre"

    noises = []

    for patient_num in patient_nums[0:]:

        img_series_path = os.path.join(os.getcwd(), "data_mri")
        img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + "pre" + "diffusion_imgs"))
        with open(img_series_path, "rb") as fp:
            img_dict_full = pickle.load(fp)
        img_dict = img_dict_full["diffusion"]

        b_values = list(img_dict.image_arrays.keys())
        b_values = np.array(b_values)
        for roi in ["r_par", "l_par"]:
            try:
                
                noise = getattr(img_dict, f"noise_{roi}")
                noises.extend(noise[:,:,:].flatten())
            except:
                continue
    noises= np.array(noises)


    num_noise = noises[noises > 0.1].size
    noise_percent = num_noise / noises.size
    fig, ax = plt.subplots()
    ax.hist(noises[noises > 0.1], bins=30, density=True, alpha=0.8, color='salmon', edgecolor="orangered", label='Noise')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Relative Frequency")
    plt.legend()
    plt.show()
    return
    
def get_curvature(b_values, params, func_type="biexp"):
    if func_type=="biexp":
        second_der = biexp_second_deriv(b_values, *params)*1000
        der = biexp_deriv(b_values, *params)*1000

        curvature =  second_der/ (1+der**2)**1.5

    elif func_type =="triexp":
        second_der = triexp_second_deriv(b_values, *params)*1000
        der = triexp_deriv(b_values, *params)*1000

        curvature =  second_der/ (1+der**2)**1.5
    return curvature
def get_auc_triexp(pD1, pD2, f1, f2, D=0.001):
    b_values= np.linspace(0,1000, 1001)
    vals = triexp(b_values, D, pD1, f1, pD2, f2)
    auc = np.sum(vals)


    return auc

def get_auc_biexp(pD, f, D=0.001):
    b_values= np.linspace(0,1000, 1001)
    vals = biexp(b_values, pD,D,f)
    auc = np.sum(vals)

    return auc

def get_b_of_max_curvature_biexp(pD, f, D=0.001):
    b_values= np.linspace(0,1000, 1001)
    # voxel = biexp(b_values, 0.018, 0.002, 0.05 )
    # plot_signal_curve(b_values, voxel)

    curvature = get_curvature(b_values, [pD, f, D], func_type="biexp")
    # plt.plot(b_values, curvature)
    # plt.show()
    max_idx = np.argmax(curvature)
    max_b = b_values[max_idx]
    return max_b 
def get_b_of_max_curvature_triexp(pD1, pD2, f1, f2, D=0.001):
    b_values= np.linspace(0,1000, 1001)

    curvature = get_curvature(b_values, [pD1, pD2, f1, f2, D], func_type="triexp")
    # plt.plot(b_values, curvature)
    # plt.show()
    max_idx = np.argmax(curvature)
    max_b = b_values[max_idx]
    return max_b 
def plot_auc_vs_params(num_bins=300, func="triexp"):
    #this function will solve for the b value of maximum curvature as a function of D*, D, and then plot? 
    D = 0.00075
    if func=="biexp":
        f_vals = list(np.linspace(0.05,0.3, num_bins))
        pD_vals = list(np.linspace(0.005,0.08, num_bins))

        b_map = np.zeros((num_bins,num_bins))
        for p_idx, pD in enumerate(pD_vals):
            for f_idx, f in enumerate(f_vals):
                max_curv_b = get_auc_biexp(pD, f, D)
                b_map[p_idx, f_idx] = max_curv_b

        fig, ax = plt.subplots(figsize=(30,30))
        im = ax.imshow(b_map, cmap="jet")
        ax.set_xlabel("$f$", fontsize=30)
        ax.set_ylabel("$D*$", fontsize=30)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='both', labelsize=20)

        x_ticks = np.linspace(0, len(f_vals), 8)
        x_tick_labels = [f'{val/len(f_vals)*(f_vals[-1]-f_vals[0]) + f_vals[0]:.2f}' for val in x_ticks]
        y_ticks = np.linspace(0, len(pD_vals), 8)
        y_tick_labels = [f'{val/len(pD_vals)*(pD_vals[-1]-pD_vals[0])+pD_vals[0]:.3f}' for val in y_ticks]

        ax.set_xticks(x_ticks, x_tick_labels)
        ax.set_yticks(y_ticks, y_tick_labels)
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=15)
        plt.show()

    elif func=="triexp":
        f_vals = list(np.linspace(0.05,0.2, num_bins))

        pD1 = 0.012
        pD2 = 0.005

        b_map = np.zeros((num_bins,num_bins))
        for f1_idx, f1 in enumerate(f_vals):
            for f2_idx, f2 in enumerate(f_vals):
                max_curv_b = get_auc_triexp(pD1,pD2, f1,f2, D)
                b_map[f1_idx, f2_idx] = max_curv_b

        fig, ax = plt.subplots(figsize=(30,30))
        im = ax.imshow(b_map, cmap="jet")
        ax.set_ylabel("$f_1$", fontsize=30)
        ax.set_xlabel("$f_2$", fontsize=30)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='both', labelsize=20)

        x_ticks = np.linspace(0, len(f_vals), 8)
        x_tick_labels = [f'{val/len(f_vals)*(f_vals[-1]-f_vals[0]) + f_vals[0]:.2f}' for val in x_ticks]
        y_ticks = np.linspace(0, len(f_vals), 8)
        y_tick_labels = [f'{val/len(f_vals)*(f_vals[-1]-f_vals[0]) + f_vals[0]:.2f}' for val in x_ticks]

        ax.set_xticks(x_ticks, x_tick_labels)
        ax.set_yticks(y_ticks, y_tick_labels)
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=15)
        plt.show()

def plot_max_curvature_location_vs_params(num_bins=300, func="biexp"):
    #this function will solve for the b value of maximum curvature as a function of D*, D, and then plot? 
    D = 0.0015
    if func=="biexp":
        f_vals = list(np.linspace(0.05,0.3, num_bins))
        pD_vals = list(np.linspace(0.005,0.08, num_bins))

        b_map = np.zeros((num_bins,num_bins))
        for p_idx, pD in enumerate(pD_vals):
            for f_idx, f in enumerate(f_vals):
                max_curv_b = get_b_of_max_curvature_biexp(pD, f, D)
                b_map[p_idx, f_idx] = max_curv_b

        fig, ax = plt.subplots(figsize=(30,30))
        im = ax.imshow(b_map, cmap="jet")
        ax.set_xlabel("$f$", fontsize=30)
        ax.set_ylabel("$D*$", fontsize=30)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='both', labelsize=20)

        x_ticks = np.linspace(0, len(f_vals), 8)
        x_tick_labels = [f'{val/len(f_vals)*(f_vals[-1]-f_vals[0]) + f_vals[0]:.2f}' for val in x_ticks]
        y_ticks = np.linspace(0, len(pD_vals), 8)
        y_tick_labels = [f'{val/len(pD_vals)*(pD_vals[-1]-pD_vals[0])+pD_vals[0]:.3f}' for val in y_ticks]

        ax.set_xticks(x_ticks, x_tick_labels)
        ax.set_yticks(y_ticks, y_tick_labels)
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=15)
        plt.show()

    elif func=="triexp":
        f_vals = list(np.linspace(0.05,0.3, num_bins))

        pD1 = 0.025
        pD2 = 0.005

        b_map = np.zeros((num_bins,num_bins))
        for f1_idx, f1 in enumerate(f_vals):
            for f2_idx, f2 in enumerate(f_vals):
                max_curv_b = get_b_of_max_curvature_triexp(pD1,pD2, f1,f2, D)
                b_map[f1_idx, f2_idx] = max_curv_b

        fig, ax = plt.subplots(figsize=(30,30))
        im = ax.imshow(b_map, cmap="jet")
        ax.set_ylabel("$f_1$", fontsize=30)
        ax.set_xlabel("$f_2$", fontsize=30)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='both', labelsize=20)

        x_ticks = np.linspace(0, len(f_vals), 8)
        x_tick_labels = [f'{val/len(f_vals)*(f_vals[-1]-f_vals[0]) + f_vals[0]:.2f}' for val in x_ticks]
        y_ticks = np.linspace(0, len(f_vals), 8)
        y_tick_labels = [f'{val/len(f_vals)*(f_vals[-1]-f_vals[0]) + f_vals[0]:.2f}' for val in x_ticks]

        ax.set_xticks(x_ticks, x_tick_labels)
        ax.set_yticks(y_ticks, y_tick_labels)
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=15)
        plt.show()


def get_image_quality_score(img_type="orig", ind=None):
    patient_nums = os.listdir(data_folder)
    patient_nums.sort()
    brisque_scores_total = []
    brisque_scores_total_orig = []

    clip_scores_total = []
    clip_scores_total_orig = []
    for patient_num in ["10", "11", "12", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']:
        img_series_path = os.path.join(os.getcwd(), "data_mri")
        img_series_path = os.path.join(img_series_path, str("SIVIM" + patient_num + "pre" + "diffusion_imgs"))
        with open(img_series_path, "rb") as fp:
            img_dict_full = pickle.load(fp)
        img_dict = img_dict_full["diffusion"]  
        for roi in ["l_par", "r_par"]:#, "cord"]:
            print(f"{patient_num}, roi {roi}")
            
            try:
    
                if img_type == "orig":
                    img = np_to_torch(getattr(img_dict, f"deconv_array_{roi}")).to('cuda')

                    img_orig = np_to_torch(getattr(img_dict, f"img_{roi}")).to('cuda')
                    img = torch.clamp(img / torch.max(img_orig), min=0, max=1)
                    img_orig = img_orig / torch.max(img_orig)
                elif img_type == "biexp":
                    img = np_to_torch(getattr(img_dict, f"biexp_img_{roi}_{True}_{False}_pre")[ind,:,:,:]).to('cuda')
                    img_orig = np_to_torch(getattr(img_dict, f"biexp_img_{roi}_{False}_{False}_pre")[ind,:,:,:]).to('cuda')
                    img = torch.nan_to_num(img, nan=np.nanmin(img.cpu().numpy()))
                    img_orig = torch.nan_to_num(img_orig, nan=np.nanmin(img_orig.cpu().numpy()))
                    img = img / torch.max(img)
                    img_orig = img_orig / torch.max(img_orig)
                elif img_type == "triexp":
                    img = np_to_torch(getattr(img_dict, f"triexp_img_{roi}_{True}_{False}_pre")[ind,:,:,:]).to('cuda')
                    img_orig = np_to_torch(getattr(img_dict, f"triexp_img_{roi}_{False}_{False}_pre")[ind,:,:,:]).to('cuda')
                    img = torch.nan_to_num(img, nan=np.nanmin(img.cpu().numpy()))
                    img_orig = torch.nan_to_num(img_orig, nan=np.nanmin(img_orig.cpu().numpy()))
                    img = img / torch.max(img)
                    img_orig = img_orig / torch.max(img_orig)
                elif img_type == "adc":
                    img = np_to_torch(getattr(img_dict, f"adc_img_{roi}_{True}_{False}_pre")[0,:,:,:]).to('cuda')
                    img_orig = np_to_torch(getattr(img_dict, f"adc_img_{roi}_{False}_{False}_pre")[0,:,:,:]).to('cuda')
                    img = torch.nan_to_num(img, nan=np.nanmin(img.cpu().numpy()))
                    img_orig = torch.nan_to_num(img_orig, nan=np.nanmin(img_orig.cpu().numpy()))

                    img = torch.clamp(img, min=0.0002, max=0.0025)
                    img_orig = torch.clamp(img_orig, min=0.0002, max=0.0025)
                    img = img / torch.max(img)
                    img_orig = img_orig / torch.max(img_orig)
                    
                elif img_type == "auc":
                    img = np_to_torch(getattr(img_dict, f"auc_s0_img_{roi}_{True}_{False}_{True}_pre")).to('cuda')
                    img_orig = np_to_torch(getattr(img_dict, f"auc_s0_img_{roi}_{False}_{False}_{True}_pre")).to('cuda')
                    img = torch.nan_to_num(img, nan=np.nanmin(img.cpu().numpy()))
                    
                    img_orig = torch.nan_to_num(img_orig, nan=np.nanmin(img_orig.cpu().numpy()))

                    img = torch.clamp(img, min=400, max=900)
                    img_orig = torch.clamp(img_orig, min=400, max=900)
                    img = img / torch.max(img)
                    img_orig = img_orig / torch.max(img_orig)
                    
                elif img_type == "auc_l":
                    img = np_to_torch(getattr(img_dict, f"auc_s0_l_img_{roi}_{True}_{False}_{True}_pre")).to('cuda')
                    img_orig = np_to_torch(getattr(img_dict, f"auc_s0_l_img_{roi}_{False}_{False}_{True}_pre")).to('cuda')
                    img = torch.nan_to_num(img, nan=np.nanmin(img.cpu().numpy()))
                    img_orig = torch.nan_to_num(img_orig, nan=np.nanmin(img_orig.cpu().numpy()))

                    img = torch.clamp(img, min=90, max=120)
                    img_orig = torch.clamp(img_orig, min=90, max=120)

                    img = img / torch.max(img)
                    img_orig = img_orig / torch.max(img_orig)
                elif img_type == "auc_m":
                    img = np_to_torch(getattr(img_dict, f"auc_s0_m_img_{roi}_{True}_{False}_{True}_pre")).to('cuda')
                    img_orig = np_to_torch(getattr(img_dict, f"auc_s0_m_img_{roi}_{False}_{False}_{True}_pre")).to('cuda')
                    img = torch.nan_to_num(img, nan=np.nanmin(img.cpu().numpy()))
                    img_orig = torch.nan_to_num(img_orig, nan=np.nanmin(img_orig.cpu().numpy()))

                    img = torch.clamp(img, min=10, max=230)
                    img_orig = torch.clamp(img_orig, min=10, max=230)

                    img = img / torch.max(img)
                    img_orig = img_orig / torch.max(img_orig)
                elif img_type == "auc_h":
                    img = np_to_torch(getattr(img_dict, f"auc_s0_h_img_{roi}_{True}_{False}_{True}_pre")).to('cuda')
                    img_orig = np_to_torch(getattr(img_dict, f"auc_s0_h_img_{roi}_{False}_{False}_{True}_pre")).to('cuda')
                    img = torch.nan_to_num(img, nan=np.nanmin(img.cpu().numpy()))
                    img_orig = torch.nan_to_num(img_orig, nan=np.nanmin(img_orig.cpu().numpy()))

                    img = torch.clamp(img, min=200, max=600)
                    img_orig = torch.clamp(img_orig, min=200, max=600)

                    img = img / torch.max(img)
                    img_orig = img_orig / torch.max(img_orig)
                
            except:
                print(f"Could not load the deblurred image data for the left side for patient {patient_num}")
                continue

            clip_iqa = CLIPIQA(data_range=1.).to('cuda')
            brisque_scores = []
            brisque_scores_orig = []

            clip_scores = []
            clip_scores_orig = []
            if img_type == "orig":
                for b in range(16):
                    im = img[0,b,:,:,:]
                    im_orig = img_orig[0,b,...]

                    for slice in range(6,im.shape[0]-6):

                        try:
                            clip_score = clip_iqa(im[slice,:,:].unsqueeze(0).unsqueeze(0))
                            clip_scores.append(clip_score.item())
                        except: continue
                    for slice in range(6,im_orig.shape[0]-6):
                        try:
                            clip_score = clip_iqa(im_orig[slice,:,:].unsqueeze(0).unsqueeze(0))
                            clip_scores_orig.append(clip_score.item())
                        except: continue
                    clip_scores_total.append(np.mean(clip_scores))    
                    clip_scores_total_orig.append(np.mean(clip_scores_orig))
                    

                    im = im.cpu()
                    im_orig = im_orig.cpu()
                    for slice in range(6,im.shape[0]-6):
                        try:
                            brisque_score = brisque(im[slice,:,:].unsqueeze(0).unsqueeze(0))
                            brisque_scores.append(brisque_score)
                        except: continue
                    for slice in range(6,im_orig.shape[0]-6):
                        try:
                            brisque_score = brisque(im_orig[slice,:,:].unsqueeze(0).unsqueeze(0))
                            brisque_scores_orig.append(brisque_score)
                        except: continue
                    brisque_scores_total.append(np.mean(brisque_scores))    
                    brisque_scores_total_orig.append(np.mean(brisque_scores_orig))
            else:
                for slice in range(6,img.shape[1]-6):

                    try:
                        clip_score = clip_iqa(img[:,slice,:,:].unsqueeze(0))
                        clip_scores.append(clip_score.item())
                    except: continue
                for slice in range(6,img_orig.shape[1]-6):
                    try:
                        clip_score = clip_iqa(img_orig[:,slice,:,:].unsqueeze(0))
                        clip_scores_orig.append(clip_score.item())
                    except: continue
                clip_scores_total.append(np.mean(clip_scores))    
                clip_scores_total_orig.append(np.mean(clip_scores_orig))
                

                img = img.cpu()
                img_orig = img_orig.cpu()
                for slice in range(6,img.shape[1]-6):
                    try:
                        brisque_score = brisque(img[:,slice,:,:].unsqueeze(0))
                        brisque_scores.append(brisque_score)
                    except: continue
                for slice in range(6,img_orig.shape[1]-6):
                    try:
                        brisque_score = brisque(img_orig[:,slice,:,:].unsqueeze(0))
                        brisque_scores_orig.append(brisque_score)
                    except: continue
                brisque_scores_total.append(np.mean(brisque_scores))    
                brisque_scores_total_orig.append(np.mean(brisque_scores_orig))
            
    mean_brisque = np.nanmean(brisque_scores_total)
    std_brisque = np.nanstd(brisque_scores_total)
    
    mean_brisque_orig = np.nanmean(brisque_scores_total_orig)
    std_brisque_orig = np.nanstd(brisque_scores_total_orig)
    print(mean_brisque, std_brisque)
    print(mean_brisque_orig, std_brisque_orig)

    _, p_brisque = ttest_rel(brisque_scores_total, brisque_scores_total_orig)

    mean_clip = np.nanmean(clip_scores_total)
    std_clip = np.nanstd(clip_scores_total)
    
    mean_clip_orig = np.nanmean(clip_scores_total_orig)
    std_clip_orig = np.nanstd(clip_scores_total_orig)

    _, p_clip = ttest_rel(clip_scores_total, clip_scores_total_orig)

    print(mean_clip, std_clip)
    print(mean_clip_orig, std_clip_orig)
    return


def plot_signal_curve(b_vals, voxel, fit_params=None):
    fig, ax = plt.subplots(figsize=(20,20))
    voxel = np.array(deepcopy(voxel)) / voxel[0]
    ax.scatter(b_vals, voxel)
    ax.set_xlabel("b value", fontsize=20)
    ax.set_ylabel(f"$S(b)/S(0)$", fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if fit_params is not None:

        #get curve from fit params 
        b_val_linspace = np.linspace(0,1000, 1000)
        fit_range = biexp(b_val_linspace, *fit_params)
        ax.plot(b_val_linspace, fit_range, c="red")
    plt.show()

def biexp_second_deriv(b, Dp, f,D):
    return Dp**2*f*np.exp(-b*Dp) + D**2*(1-f)*np.exp(-b*D)

def biexp_deriv(b, Dp, f,D):
    return -Dp*f*np.exp(-b*Dp) - D*(1-f)*np.exp(-b*D)

def triexp_second_deriv(b, Dp1, Dp2, f1, f2, D):
    return Dp1**2*f1*np.exp(-b*Dp1) + Dp2**2*f2*np.exp(-b*Dp2) +  D**2*(1-f1-f2)*np.exp(-b*D)

def triexp_deriv(b, Dp1, Dp2, f1, f2, D):
    return -Dp1*f1*np.exp(-b*Dp1) - Dp2*f2*np.exp(-b*Dp2) -  D*(1-f1-f2)*np.exp(-b*D)

def biexp(b, Dp, D, f):
    return f*np.exp(-b*Dp) + (1-f)*np.exp(-b*D)

def triexp(b, D, Dp1, f1, Dp2, f2):
    return f1*np.exp(-b*Dp1) + f2*np.exp(-b*Dp2) + (1-f1-f2)*np.exp(-b*D)

def plot_diffusion_img_with_signal_curve(scan_type="pre"):
    b_vals = [0,20,30,40,50,60,70,80,90,100,120,150,250,400,800,1000]
    for patient in ["10","11" "12", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']:
        for roi in ["r_par", "l_par"]:
            try:
                img_series_path = os.path.join(os.getcwd(), "data_mri")
                img_series_path = os.path.join(img_series_path, str("SIVIM" + patient + scan_type + "diffusion_imgs"))
                with open(img_series_path, "rb") as fp:
                    img_dict_full = pickle.load(fp)
                img_dict = img_dict_full["diffusion"]
                out_x = getattr(img_dict, f"deconv_array_{roi}")
                out_k = getattr(img_dict, f"kernel_{roi}")
                img_orig = getattr(img_dict, f"img_{roi}")

            except:
                continue
            # out_x = zoom(out_x, zoom=(1, 0.25, 0.25, 0.25), order=1)
            # img_orig = zoom(img_orig, zoom=(1, 0.25, 0.25, 0.25), order=1)
            mean = np.nanmean(img_orig[0,...])
            std = np.std(img_orig[0,...])
            #img_orig[img_orig < 150] = 0 
            #out_x = (out_x - np.nanmean(img_orig[0,...])) / np.std(img_orig[0,...])
            #img_orig = (img_orig - np.nanmean(img_orig[0,...])) / np.nanstd(img_orig[0,...])
            
            b_slice=15
            plot_3d_image(out_k)
            print(np.sum(out_k))
            current_slice = int(out_x.shape[1]/2)
            cmap='viridis'
            def update_slide(val):
                slice_index = int(slider.val)
                #global current_slice
                current_slice = slice_index
                
                im = ax[0].imshow(out_x[b_slice,slice_index, :,:], cmap=cmap, vmin=np.nanmin(img_orig[b_slice,val,:,:]), vmax=np.nanmax(img_orig[b_slice,val,:,:]))
                im2 = ax[1].imshow(img_orig[b_slice,slice_index, :,:], cmap=cmap, vmin=np.nanmin(img_orig[b_slice,val,:,:]), vmax=np.nanmax(img_orig[b_slice,val,:,:]))
                fig.canvas.draw_idle()
            def update_curve(sel):
                #global current_slice
                if sel.target.index is not None:
                    voxel_indices = sel.target.index
                    sigs = (out_x[:,current_slice, voxel_indices[0], voxel_indices[1]]) / (out_x[0,current_slice, voxel_indices[0], voxel_indices[1]])
                    ax[2].cla()
                    ax[2].scatter(b_vals, (sigs), c='r')

                    sigs_orig = (img_orig[:, current_slice, voxel_indices[0], voxel_indices[1]]) / (img_orig[0, current_slice, voxel_indices[0], voxel_indices[1]])
                    ax[3].cla()
                    ax[3].scatter(b_vals, (sigs_orig), c='r')
                    ax[2].set_title("Deblurred Voxel")
                    ax[3].set_title("Original Voxel")
                    ax[2].set_xlabel("b-value ($s/mm^2$)")
                    ax[2].set_ylabel("Signal (normalized)")
                    ax[3].set_xlabel("b-value ($s/mm^2$)")
                    ax[3].set_ylabel("Signal (normalized)")

                
            
            fig, ax = plt.subplots(1,4,figsize=(21,7))
            im = ax[0].imshow(out_x[b_slice,int(out_x.shape[1]/2), :,:], cmap=cmap, vmin=np.nanmin(img_orig[b_slice,int(out_x.shape[1]/2), :,:]), vmax=np.nanmax(img_orig[b_slice,int(out_x.shape[1]/2), :,:]))
            im2 = ax[1].imshow(img_orig[b_slice,int(out_x.shape[1]/2), :,:], cmap=cmap, vmin=np.nanmin(img_orig[b_slice,int(out_x.shape[1]/2), :,:]), vmax=np.nanmax(img_orig[b_slice,int(out_x.shape[1]/2), :,:]))
            ax[2].scatter([],[])
            ax[3].scatter([],[])
            #colorbar = plt.colorbar(im, ax=ax)
            ax_slider = plt.axes([0.2,0.01,0.65,0.03], facecolor='green')
            slider = Slider(ax=ax_slider, label="Slice", valmin=0, valmax=out_x.shape[1]-1, valstep=1, valinit=int(out_x.shape[1]/2))
            slider.on_changed(update_slide)
            ax[0].set_title("Deblurred")
            ax[1].set_title("Original")
            ax[2].set_title("Deblurred Voxel")
            ax[3].set_title("Original Voxel")
            ax[0].spines['top'].set_visible(False)
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['bottom'].set_visible(False)
            ax[0].spines['left'].set_visible(False)

            # Turn off all ticks
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[1].spines['top'].set_visible(False)
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['bottom'].set_visible(False)
            ax[1].spines['left'].set_visible(False)

            # Turn off all ticks
            ax[1].set_xticks([])
            ax[1].set_yticks([])

            cursors = mplcursors.cursor(im, hover=False, highlight=False)
            cursors.connect('add', update_curve)
            plt.show(block=True)
            plt.close("all")
    
def get_performance_stats():
    patient_nums = os.listdir(data_folder)
    patient_nums.sort()

    mse_vals = []
    mae_vals = []
    doses_pred = []
    doses_orig = []
    doses_diff = []
    for patient in patient_nums[:]:
        with open(os.path.join(os.getcwd(),"models",  f"{patient}_model_stuff.txt"), "rb") as fp:
            [_,_, pet_img_view, prediction, _, loss_history, step] = pickle.load(fp)
        with open(os.path.join(os.getcwd(),"predictions",  f"{patient}_registered_predictions.txt"), "rb") as fp:
            out_x, out_k = pickle.load(fp)   
        if torch.is_tensor(out_k):    
            out_k = out_k.squeeze().detach().cpu().numpy()
            pet_img_view = pet_img_view.squeeze().detach().cpu().numpy()
        mae = np.mean(np.abs(pet_img_view-prediction))
        rmse = np.sqrt(np.mean((pet_img_view-prediction)**2))
        mae_vals.append(mae)
        mse_vals.append(rmse)
        doses_orig.append(np.sum(pet_img_view))
        doses_pred.append(np.sum(prediction))    
        doses_diff.append(np.abs(np.sum(pet_img_view)-np.sum(prediction))/np.sum(pet_img_view))
    print(f"avg mae: {np.mean(mae_vals)} +- {np.std(mae_vals)}")
    print(f"avg rmse: {np.mean(mse_vals)} +- {np.std(mse_vals)}")
    print(f"avg dose diff: {np.mean(doses_diff)} +- {np.std(doses_diff)}")
    return 

def get_gland_stats(deblurred=False, load=True):
    par_voxels = {}
    sm_voxels = {}
    patient_nums = os.listdir(data_folder)
    patient_nums.sort()
    if load == True:
        try:
            with open(os.path.join(os.getcwd(), "cache", f"gland_stats_dict_{deblurred}"), "rb") as fp:
                par_voxels, sm_voxels = pickle.load(fp)
        except:
            print("Could not load the gland stats...")      
    else:        
        for patient_num in patient_nums:
            par_voxels[patient_num] = []
            sm_voxels[patient_num] = []

            print(f"Loading data for for {patient_num}...")
            
            img_series_path = os.path.join(os.getcwd(), "data", patient_num, "img_dict")
            with open(img_series_path, "rb") as fp:
                img_dict = pickle.load(fp)
            with open(os.path.join(os.getcwd(), "data", patient_num, "mask_dict"), "rb") as fp:
                mask_dict = pickle.load(fp)["PET"]
            if deblurred==True:
                with open(os.path.join(os.getcwd(),"predictions",  f"{patient_num}_registered_predictions.txt"), "rb") as fp:
                    image, _ = pickle.load(fp)

                    coords_array = img_dict["PET"].deconv_coords_2x
            elif deblurred==False:
                image = img_dict["PET"].image_array * img_dict["PET"].suv_factors[1]
                image = zoom(image, zoom=(2,2,2), order=0)
                coords_array = np.zeros((img_dict["PET"].coords_array.shape[0],img_dict["PET"].coords_array.shape[1]*2,img_dict["PET"].coords_array.shape[2]*2,img_dict["PET"].coords_array.shape[3]*2))
                coords_array[0,:,:,:] = zoom(img_dict["PET"].coords_array[0,:,:,:], zoom=2, order=1)
                coords_array[1,:,:,:] = zoom(img_dict["PET"].coords_array[1,:,:,:], zoom=2, order=1)
                coords_array[2,:,:,:] = zoom(img_dict["PET"].coords_array[2,:,:,:], zoom=2, order=1)
                #coords_array = img_dict["PET"].coords_array
            

            #now get the parotid / submandibular gland stats
            for structure in mask_dict:
                contours = mask_dict[structure].wholeROI
                mask = get_contour_masks.get_contour_masks(contours, coords_array)
                voxel_vals = deepcopy(image[mask])
                if "sm" in structure.lower():
                    sm_voxels[patient_num].extend(voxel_vals)
                elif "par" in structure.lower():
                    par_voxels[patient_num].extend(voxel_vals)


        with open(os.path.join(os.getcwd(), "cache", f"gland_stats_dict_{deblurred}"), "wb") as fp:
            pickle.dump([par_voxels, sm_voxels], fp)

    par_means = []
    par_maxs = []
    sm_means = []
    sm_maxs = []
    for patient in par_voxels:
        par_means.append(np.mean(par_voxels[patient]))
        par_maxs.append(np.max(par_voxels[patient]))
        sm_means.append(np.mean(sm_voxels[patient]))
        sm_maxs.append(np.max(sm_voxels[patient]))
    
    par_mean_mean = np.mean(par_means)
    par_mean_max = np.mean(par_maxs)
    sm_mean_mean = np.mean(sm_means)
    sm_mean_max = np.mean(sm_maxs)

    par_mean_mean_std = np.std(par_means)
    par_mean_max_std = np.std(par_maxs)
    sm_mean_mean_std = np.std(sm_means)
    sm_mean_max_std = np.std(sm_maxs)


    print(f"par mean mean: {par_mean_mean} +- {par_mean_mean_std}")
    print(f"par mean max: {par_mean_max}+- {par_mean_max_std}")
    print(f"sm mean mean: {sm_mean_mean} +- {sm_mean_mean_std}")
    print(f"sm mean max: {sm_mean_max} +- {sm_mean_max_std}")

    #make combined histogram
    bin_edges = np.linspace(0, 25, 21)
    hist_avg = np.zeros(20)
    plt.figure(figsize=(10, 7))
    for p, patient in enumerate(sm_voxels):
        hist, _ = np.histogram(sm_voxels[patient], bins=bin_edges)
        hist = np.array(hist) / len(sm_voxels[patient])
        hist_avg += hist
    hist_avg /= 30

    # Step 5: Plot the average histogram
    plt.figure(figsize=(24, 18))
    plt.bar(bin_edges[:-1], hist_avg, width=np.diff(bin_edges), align='edge', color="darkred", edgecolor='k')
    plt.xlabel("$SUV_{lbm}$", fontsize=20)
    plt.ylabel("Mean proportion", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)

    #plt.grid(True)
    plt.show(block=True)

    return 

def plot_predicted_kernels(cmap="plasma"):
    with open(os.path.join(os.getcwd(), "misc", f"spacing_types.txt"), "rb") as fp:
        spacing_types = pickle.load(fp)   
    patient_nums = os.listdir(data_folder)
    font_size = 12
    patient_nums.sort()
    for patient in patient_nums[:]:
        with open(os.path.join(os.getcwd(),"models",  f"{patient}_model_stuff.txt"), "rb") as fp:
            [_,_, pet_img_view, prediction, _, loss_history, step] = pickle.load(fp)
        with open(os.path.join(os.getcwd(),"predictions",  f"{patient}_registered_predictions.txt"), "rb") as fp:
            out_x, out_k = pickle.load(fp)   
        if torch.is_tensor(out_k):    
            out_k = out_k.squeeze().detach().cpu().numpy()
            pet_img_view = pet_img_view.squeeze().detach().cpu().numpy()


        orig_slice = int(pet_img_view.shape[0] * 0.68)
        deblur_slice = orig_slice * 2    

        fig, axs = plt.subplots(1,4, figsize=(25,7))
        axs[3].imshow(pet_img_view[orig_slice,:,:], cmap=cmap, vmax=np.amax(out_x[deblur_slice,:,:]))
        axs[3].set_xticks([])
        axs[3].set_yticks([])
        axs[3].set_title("Original Image, $y$", fontsize=font_size+3)


        im_2 = axs[0].imshow(out_x[deblur_slice,:,:], cmap=cmap, vmax=np.amax(out_x[deblur_slice,:,:]))
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_title("Deblurred Image, $x$", fontsize=font_size+3)

        im_3 = axs[1].imshow(out_k[7,:,:], cmap=cmap, vmax=np.amax(out_k[7,:,:]))
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_title("Kernel, $k$", fontsize=font_size+3)

        im_2 = axs[2].imshow(prediction[orig_slice,:,:], cmap=cmap, vmax=np.amax(out_x[deblur_slice,:,:]))
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        axs[2].set_title("$x * k$", fontsize=font_size+3)
        plt.tight_layout()
        plt.show()
        print(patient)

def plot_loss_history():
    patient_nums = os.listdir(data_folder)
    patient_nums.sort()
    for patient in patient_nums[7:]:

        #get pre_loss
        with open(os.path.join(os.getcwd(),"models",  f"{patient}_pre_loss.txt"), "rb") as fp:
            pre_loss = pickle.load(fp)
        with open(os.path.join(os.getcwd(),"models",  f"{patient}_model_stuff.txt"), "rb") as fp:
            [_,_, _,_, _, loss_history, step] = pickle.load(fp)
        print(patient)
        pre_loss.extend(loss_history)
        plt.plot(range(-2000,5000), np.log(pre_loss), color="r")
        #plt.ylim(0,0.004)
        plt.ylabel("ln (loss)", fontsize=16)
        plt.xlabel("Iteration", fontsize=16)
        plt.show()
        
        

    return

def compare_predicted_kernels():
  
    
    combined_kernel_image = np.zeros([11,11,11])
    patient_nums = ["10", "11","12", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']
    kernels = {}
    maxs = []
    patient_region_inner_prods = []
    for patient in patient_nums:
        img_series_path = os.path.join(os.getcwd(), "data_mri")
        img_series_path = os.path.join(img_series_path, str("SIVIM" + patient + "pre" + "diffusion_imgs"))

        with open(os.path.join(os.path.join(os.getcwd(), "data_mri"), str("sivim" + patient + "_" + "pre" + "_MR_mask_dict")), "rb") as fp:
            mask_dict = pickle.load(fp)
        patient_kernels = [] #store the predicted kernels
        with open(img_series_path, "rb") as fp:
            img_dict_full = pickle.load(fp)
        img_dict = img_dict_full["diffusion"] 
        for roi in ["l_par", "r_par"]:
            if roi not in mask_dict:
                continue
            try:         
                kernel = getattr(img_dict, f"kernel_{roi}")
                patient_kernels.append(kernel)
                maxs.append(np.amax(kernel))
            except:
                continue

        #get avg kernel for patient (over rois)
        avg_kern = patient_kernels[0]
        #plot_3d_image(avg_kern)

        for i in range(1, len(patient_kernels)):
            shift, error, diff = phase_cross_correlation(patient_kernels[0], patient_kernels[i])
            kernel_2 = shifter(patient_kernels[i], [val for val in shift], order=1)
            kernel_2 /= np.sum(kernel_2)
            avg_kern += kernel_2

        avg_kern /= np.sum(avg_kern)
        kernels[patient] = avg_kern
        #plot_3d_image(avg_kern)

        #also need the mean of inner products between kernels:
        inner_prods_patient = []
        for i in range(len(patient_kernels)):
            for j in range(len(patient_kernels)):
                if i >= j:
                    continue
                kernel_1 = deepcopy(patient_kernels[i]) 
                kernel_2 = deepcopy(patient_kernels[j]) 
                shift, error, diff = phase_cross_correlation(kernel_1, kernel_2)
                #use middle slice of both images .

                kernel_2 = shifter(kernel_2, [val for val in shift], order=1)
                
                norm_1 = np.sqrt(np.sum(kernel_1**2))
                norm_2 = np.sqrt(np.sum(kernel_2**2))
                kernel_1  /= norm_1
                kernel_2  /= norm_2
                inner_product = np.sum(kernel_1 * kernel_2)
                inner_prods_patient.append(inner_product)
        if len(inner_prods_patient) > 0:
            inner_prod_patient = np.mean(inner_prods_patient)
            patient_region_inner_prods.append(inner_prod_patient)
    mean_max = np.mean(maxs)
    max_max = np.amax(maxs)
    min_max = np.amin(maxs)
    mean_patient_inner_prod = np.mean(patient_region_inner_prods)
    std_patient_inner_prod = np.std(patient_region_inner_prods)
    #first centre the first kernel
    centre_img = np.zeros((11,11,11))
    centre_img[5,5,5] = 1
    shift, error, diff = phase_cross_correlation(centre_img, kernels["10"])
    kernel = shifter(kernels["10"], [val for val in shift], order=1)
    kernels["10"] = kernel / np.sum(kernel)

    #register kernels 
    for patient in ["10", "11", "12", "13", "15", "16", '17', '18', '19', '20', '21', '22', '23']:
        shift, error, diff = phase_cross_correlation(kernels["10"], kernels[patient])
        kernel = shifter(kernels[patient], [val for val in shift], order=1)
        kernels[patient] = kernel / np.sum(kernel)

    num_kernels = len(kernels)
    inner_products = np.zeros((num_kernels,num_kernels))
    kernel_vals = []




    #metrics to list
    max_vals = []
    vals_anterior = []
    vals_posterior = []
    vals_superior = []
    vals_inferior = []
    vals_left = []
    vals_right =[ ]
    for k1, kern_1 in enumerate(list(kernels.keys())):
        kernel_1 = kernels[kern_1]
        max_vals.append(np.amax(kernel_1))
        vals_anterior.append(np.sum(kernel_1[:,0:5,:])+np.sum(kernel_1[:,5,:])/2)
        vals_posterior.append(np.sum(kernel_1[:,6:,:])+np.sum(kernel_1[:,5,:])/2)
        vals_inferior.append(np.sum(kernel_1[0:5,:,:])+np.sum(kernel_1[5,:,:])/2)
        vals_superior.append(np.sum(kernel_1[6:,:,:])+np.sum(kernel_1[5,:,:])/2)
        vals_left.append(np.sum(kernel_1[:,:,0:5])+np.sum(kernel_1[:,:,5])/2)
        vals_right.append(np.sum(kernel_1[:,:,6:])+np.sum(kernel_1[:,:,5])/2)

        combined_kernel_image += kernel_1
        for k2, kern_2 in enumerate(list(kernels.keys())):  
            # if int(kern_2) > int(kern_1):
            #     continue
            if kern_1 == kern_2:
                inner_products[int(k1)-1, int(k1)-1] = 1
                continue

            aligned_kernel_2 = deepcopy(kernels[kern_2])
            #now get the inner product (maximum correlation. Also, translate kern_2 to align with kern_1)

            

            # cross_corr = correlate(kernel_1, kernel_2)
            # print(kern_1, kern_2)
            # max_corr_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
            # shift = np.array(max_corr_idx) - np.array(kernel_1.shape) + 1
            
            # aligned_kernel_2 = np.zeros_like(kernel_2)
            # for z in range(kernel_2.shape[0]):
            #     for y in range(kernel_2.shape[1]):
            #         for x in range(kernel_2.shape[2]):
            #             aligned_kernel_2[z,y,x] = kernel_2[(z-shift[0])%kernel_2.shape[0],(y-shift[1])%kernel_2.shape[1],(x-shift[2])%kernel_2.shape[2] ]

            #first nromalize kernels
            norm_1 = np.sqrt(np.sum(kernel_1 **2))
            norm_2 = np.sqrt(np.sum(aligned_kernel_2**2))
            kernel_1 /= norm_1
            aligned_kernel_2 /= norm_2
            inner_product = np.sum(kernel_1 * aligned_kernel_2)
            inner_products[k1, k2] = inner_product
            if k1 > k2:
                kernel_vals.append(inner_product)
    combined_kernel_image /= len(list(kernels.keys()))
    #plot_3d_image(combined_kernel_image)
    plot_2d_image(np.sum(combined_kernel_image, axis=0)) 
    plot_2d_image(np.sum(combined_kernel_image, axis=1)) 
    plot_2d_image(np.sum(combined_kernel_image, axis=2)) 
    mean = np.mean(kernel_vals)
    print(mean)
    std = np.std(kernel_vals)
    print(std)
    #make all the unfilled slots nan to not be plotted with color
    inner_products[inner_products == 0] = np.nan

    mean_max = np.mean(max_vals)
    mean_ant = np.mean(vals_anterior) 
    mean_post =  np.mean(vals_posterior) 
    mean_sup = np.mean(vals_superior) 
    mean_inf = np.mean(vals_inferior) 
    mean_left = np.mean(vals_left)
    mean_right = np.mean(vals_right)



    std_max = np.std(max_vals)
    std_ant = np.std(vals_anterior) 
    std_post =  np.std(vals_posterior) 
    std_sup = np.std(vals_superior) 
    std_inf = np.std(vals_inferior) 
    std_left = np.std(vals_left)
    std_right = np.std(vals_right)

    ttest_ap = ttest_rel(vals_anterior, vals_posterior)
    ttest_si = ttest_rel(vals_superior, vals_inferior)
    ttest_ml = ttest_rel(vals_left, vals_right)

    print(f"max: {mean_max} +- {std_max}")
    print(f"ant: {mean_ant} +- {std_ant}")
    print(f"pos: {mean_post} +- {std_post}")
    print(f"sup: {mean_sup} +- {std_sup}")
    print(f"inf: {mean_inf} +- {std_inf}")
    print(f"left: {mean_left} +- {std_left}")
    print(f"right: {mean_right} +- {std_right}")



    #now make a heat map of kernel inner products
    fig, ax = plt.subplots(figsize=(10,10))
    heatmap = ax.imshow(inner_products, cmap="turbo", vmin=0.5)
    cbar = plt.colorbar(heatmap, ax=ax, shrink=0.8)
    ax.set_xlabel("Patient", fontsize=16)
    ax.set_ylabel("Patient", fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    cbar.ax.tick_params(labelsize=14) 
    ax.set_title('Inner Product Heatmap')
    plt.show()

    return


def get_patient_voxel_stats():
    spacing_types = {} #this cohort is either slice thickness and pixel spacing = [2.8, 2.73] or [3.64, 3.27] so need to compare kernels accordingly 
    slice_thickness_list = []
    pixel_spacing_list = []
    combined_list = []
    data_folder = os.path.join(os.getcwd(), "data")
    patient_nums = os.listdir(data_folder)
    patient_nums.sort()
    for patient_num in patient_nums:
        print(f"Loading data for for {patient_num}...")

        img_series_path = os.path.join(os.getcwd(), "data", patient_num, "img_dict")

        with open(img_series_path, "rb") as fp:
            img_dict = pickle.load(fp)
        pixel_spacing = img_dict["PET"].pixel_spacing[0]
        slice_thickness = img_dict["PET"].slice_thickness

        if int(slice_thickness) == 3:
            spacing_types[patient_num] = 1
        else:
            spacing_types[patient_num] = 0    

        slice_thickness_list.append(slice_thickness)
        pixel_spacing_list.append(pixel_spacing)
        combined_list.append([slice_thickness, pixel_spacing])
    with open(os.path.join(os.getcwd(), "misc", f"spacing_types.txt"), "wb") as fp:
        pickle.dump(spacing_types, fp)      
    return     

def plot_original_and_deblurred(patient='12', cmap='plasma',side="right"):

    
    #this creates a 2x2 subplot of the original and deblurred image, with a line across the body, and below a plot of the uptake vs distance across the line. can then compare max gradient, etc. 
    img_series_path = os.path.join(os.getcwd(), "data_mri")
    img_series_path = os.path.join(img_series_path, str("SIVIM" + patient + "pre" + "diffusion_imgs"))
    with open(img_series_path, "rb") as fp:
        img_dict_full = pickle.load(fp)
    img_dict = img_dict_full["diffusion"]
    roi="r_par"
    img_orig = getattr(img_dict, f"img_{roi}")
    x = getattr(img_dict, f"deconv_array_{roi}")
    y = getattr(img_dict, f"deconv_array_blurred_{roi}")
    k = getattr(img_dict, f"kernel_{roi}")

    img_slice=int(x.shape[1] / 2)
    b_val = 0
    font_size=16
    fig, axs = plt.subplots(1,2, figsize=(20,20/2))
    #first plot the images with line
    axs[0].imshow(img_orig[b_val, img_slice,:,:], cmap=cmap, vmax=np.amax(x[img_slice,:,:]))
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    # axs[1].imshow(y[b_val,img_slice,:,:], cmap=cmap, vmax=np.amax(x[img_slice,:,:]))
    # axs[1].set_xticks([])
    # axs[1].set_yticks([])

    # axs[2].imshow(np.round(np.sum(k, axis=0),8), cmap=cmap)
    # axs[2].set_xticks([])
    # axs[2].set_yticks([])

    axs[1].imshow(y[b_val, img_slice,:,:], cmap=cmap, vmax=np.amax(x[img_slice,:,:]))
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    # cbar = plt.colorbar(im_2, ax=axs[1])
    # cbar.ax.tick_params(labelsize=16)
    
    plt.tight_layout()
    #plt.savefig(os.path.join(os.getcwd(), "Figures", f"{patient}_{deblur_slice}_{deblur_y_mid}.jpg"), dpi=1000)
    plt.show(block=True)
    return

def create_uptake_line_plot(patient="02", cmap='inferno'):
    #this creates a 2x2 subplot of the original and deblurred image, with a line across the body, and below a plot of the uptake vs distance across the line. can then compare max gradient, etc. 
    with open(os.path.join(os.getcwd(),"models",  f"{patient}_model_stuff.txt"), "rb") as fp:
        [out_x, out_k, pet_img_view, _, _, _, _] = pickle.load(fp)  
    with open(os.path.join(os.getcwd(), "misc", f"spacing_types.txt"), "rb") as fp:
        spacing_types = pickle.load(fp)  
    if spacing_types[patient] == 1:
        spacing=3.27
    else: 
        spacing = 2.8         
    print(spacing_types[patient])        
    big_pet = zoom(pet_img_view, zoom=2, order=1)
    from skimage.registration import phase_cross_correlation
    shift, error, diff = phase_cross_correlation(big_pet, out_x)
    #use middle slice of both images .
    out_x = shifter(out_x, [val for val in shift])
    #plot_3d_image(out_k)
    out_k = shifter(out_k, [val/2 for val in shift])
    #plot_3d_image(out_k)
    orig_slice = int(pet_img_view.shape[0] * 0.85)
    deblur_slice = orig_slice * 2
    deblur_y_mid = int(pet_img_view.shape[1] * 2 * 0.4)
    orig_y_mid = int(deblur_y_mid / 2)

    # pet_img_view = pet_img_view[:,26:52, 17:90]
    # out_x = out_x[:,52:104,34:180]
    # orig_y_mid -= 26
    # deblur_y_mid -= 52
    font_size=16
    fig, axs = plt.subplots(2,2, figsize=(10,10))
    #first plot the images with line
    axs[0,0].imshow(pet_img_view[orig_slice,:,:], cmap=cmap, vmax=np.amax(out_x[deblur_slice,:,:]))
    axs[0,0].set_xticks([])
    axs[0,0].set_yticks([])
    axs[0,0].axhline(y=orig_y_mid, color='red', linestyle='-')
    axs[0,0].set_title("Original", fontsize=font_size+3)

    im_2 = axs[0,1].imshow(out_x[deblur_slice,:,:], cmap=cmap, vmax=np.amax(out_x[deblur_slice,:,:]))
    axs[0,1].set_xticks([])
    axs[0,1].set_yticks([])
    axs[0,1].axhline(y=deblur_y_mid, color='red', linestyle='-')
    axs[0,1].set_title("Improved", fontsize=font_size+3)
    # cbar = plt.colorbar(im_2, ax=axs[0,1])
    # cbar.ax.tick_params(labelsize=16)
    #add the line graphs
    axs[1,0].plot(np.arange(pet_img_view.shape[2]), pet_img_view[orig_slice, orig_y_mid,:], color='r')
    axs[1,0].set_xlabel("Patient Right to Left (mm)",fontsize=font_size)
    axs[1, 0].set_ylabel('PET $SUV_{lbm}$',fontsize=font_size)
    ticks = ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x*spacing))
    tick_set = np.arange(0, pet_img_view.shape[2]-1, 10)
    axs[1,0].set_xticks(tick_set)
    axs[1,0].xaxis.set_major_formatter(ticks)
    axs[1,0].set_xlim(0, pet_img_view.shape[2]-1)
    axs[1,0].set_ylim(0)

    axs[1,1].plot(np.arange(out_x.shape[2]), out_x[deblur_slice, deblur_y_mid,:], color='r')
    axs[1,1].set_xlabel("Patient Right to Left (mm)",fontsize=font_size)
    axs[1, 1].set_ylabel('PET $SUV_{lbm}$',fontsize=font_size)
    axs[1,1].set_title("")
    ticks = ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x*spacing/2))
    tick_set = np.arange(0, out_x.shape[2]-1, 20)
    axs[1,1].set_xticks(tick_set)
    axs[1,1].xaxis.set_major_formatter(ticks)
    axs[1,1].set_xlim(0, out_x.shape[2]-1)
    axs[1,1].set_ylim(0)
    plt.tight_layout()
    #plt.savefig(os.path.join(os.getcwd(), "Figures", f"{patient}_{deblur_slice}_{deblur_y_mid}.jpg"), dpi=1000)
    plt.show(block=True)
    
    return

def register_predicted_images():
    patient_nums = os.listdir(data_folder)
    patient_nums.sort()
    for patient_num in patient_nums:
        try:
            with open(os.path.join(os.getcwd(),"models",  f"{patient_num}_model_stuff.txt"), "rb") as fp:
                [out_x, out_k, pet_img_view, prediction, ct_tex, loss_history, step] = pickle.load(fp)
        except: continue  
        big_pet = zoom(pet_img_view, zoom=2, order=1)
        shift, error, diff = phase_cross_correlation(big_pet, out_x)
        #use middle slice of both images .
        out_x = shifter(out_x, [val for val in shift])
        out_k = shifter(out_k, [val/2 for val in shift])  
        print(f"Registered and saving image and kernel for patient: {patient_num}")
        with open(os.path.join(os.getcwd(),"predictions",  f"{patient_num}_registered_predictions.txt"), "wb") as fp:
            pickle.dump([out_x, out_k], fp)
    #predicted convolution kernels and deblurred images are not necessarily centered (in-line) with the original image. this function iterates through and registers the predicted images with the originals and 
    #translates the images and kernels accordingly. 


def eigen_kernels(include_spacing_type="both"):
    cmap="viridis"
    with open(os.path.join(os.getcwd(), "misc", f"spacing_types.txt"), "rb") as fp:
        spacing_types = pickle.load(fp)   
    #use svd to represent the eigenkernels of the data for all patients. 
    patient_nums = os.listdir(data_folder)
    patient_nums.sort()
    kernels = []
    for patient in patient_nums:
        spacing_type = spacing_types[patient]
        if (type(include_spacing_type) is int and include_spacing_type != spacing_type): 
            continue
        try:
            with open(os.path.join(os.getcwd(),"predictions",  f"{patient}_registered_predictions.txt"), "rb") as fp:
                _, out_k = pickle.load(fp)
        except:
            continue
        if include_spacing_type=="both" and spacing_type==1:
            out_k = resample_kernel_spacing(out_k)
        kernels.append(out_k.flatten())

    #now make the design matrix from the flattened kernels
    x_orig = np.zeros((kernels[0].size, len(kernels)))
    for k,kernel in enumerate(kernels):
        x_orig[:,k] = kernel

    #subtract the mean from columns
    mean_kernel = np.mean(x_orig, axis=1)   
    x = x_orig - np.tile(mean_kernel, (x_orig.shape[1],1)).T

    #get scd 
    U, S, VT = np.linalg.svd(x, full_matrices=0)
    top_4 = np.sum(S[0:4]**2)/np.sum(S**2)

    fig1 = plt.figure()
    fig1.set_size_inches(w=20, h=20*10/12)
    fig1.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, wspace=0.4, hspace=0)
    gs = gridspec.GridSpec(5,12, hspace=0.05,wspace=0.5, bottom=0.3,
                       left=0.02, right=0.95)

    ax_1 = plt.subplot2grid((5,12), (0,0), colspan=4, rowspan=4)
    plt.title("Axial", fontsize=18) 
    plt.imshow(np.sum(np.reshape(mean_kernel, (15,15,15)), axis=0), cmap=cmap)
    plt.colorbar()
    ax_1.set_xlabel(r"Patient Right $\rightarrow$ Left", fontsize=16)
    ax_1.set_ylabel(r"Patient Anterior $\rightarrow$ Posterior", fontsize=16)
    ax_1.set_xticks([])
    ax_1.set_yticks([])
    #plt.axis('off')


    plt.subplot2grid((5,12), (4,0), colspan=1, rowspan=1)
    plt.imshow(np.sum(np.reshape((1-U[:,0])/np.amax(1-U[:,0]), (15,15,15)), axis=0), cmap=cmap)
    
    #plt.axis('off')
    plt.axis('off')

    plt.subplot2grid((5,12), (4,1))
    plt.imshow(np.sum(np.reshape(U[:,1]/np.amax(U[:,1]), (15,15,15)), axis=0), cmap=cmap)
    plt.axis('off')
    #plt.axis('off')
    plt.subplot2grid((5,12), (4,2))

    plt.imshow(np.sum(np.reshape(U[:,2]/np.amax(U[:,2]), (15,15,15)), axis=0), cmap=cmap)
    plt.axis('off')
    #plt.axis('off')
    
    plt.subplot2grid((5,12), (4,3))
    plt.imshow(np.sum(np.reshape(U[:,3]/np.amax(U[:,3]), (15,15,15)), axis=0), cmap=cmap)
    plt.axis('off')
    #plt.axis('off')

    #coronal
    ax_c = plt.subplot2grid((5,12), (0,4), colspan=4, rowspan=4)
    plt.title("Coronal", fontsize=18)
    ax_c.set_xlabel(r"Patient Right $\rightarrow$ Left", fontsize=16)
    ax_c.set_ylabel(r"Patient Inferior $\rightarrow$ Superior", fontsize=16)
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    plt.imshow(np.sum(np.reshape(mean_kernel, (15,15,15)), axis=1), cmap=cmap)
    


    plt.subplot2grid((5,12), (4,4), colspan=1, rowspan=1)
    plt.imshow(np.sum(np.reshape((1-U[:,0])/np.amax(1-U[:,0]), (15,15,15)), axis=1), cmap=cmap)
    plt.axis('off')


    plt.subplot2grid((5,12), (4,5))
    plt.imshow(np.sum(np.reshape(U[:,1]/np.amax(U[:,1]), (15,15,15)), axis=1), cmap=cmap)
    plt.axis('off')
    plt.subplot2grid((5,12), (4,6))

    plt.imshow(np.sum(np.reshape(U[:,2]/np.amax(U[:,2]), (15,15,15)), axis=1), cmap=cmap)
    plt.axis('off')
    
    plt.subplot2grid((5,12), (4,7))
    plt.imshow(np.sum(np.reshape(U[:,3]/np.amax(U[:,3]), (15,15,15)), axis=1), cmap=cmap)
    plt.axis('off')

    #sagittal
    ax_s = plt.subplot2grid((5,12), (0,8), colspan=4, rowspan=4)

    plt.title("Sagittal", fontsize=18)
    ax_s.set_xlabel(r"Patient Anterior $\rightarrow$ Posterior", fontsize=16)
    ax_s.set_ylabel(r"Patient Inferior $\rightarrow$ Superior", fontsize=16)
    ax_s.set_xticks([])
    ax_s.set_yticks([])
    plt.imshow(np.reshape(mean_kernel, (15,15,15))[:,:,7], cmap=cmap)



    plt.subplot2grid((5,12), (4,8), colspan=1, rowspan=1)
    plt.imshow(np.sum(np.reshape((1-U[:,0])/np.amax(1-U[:,0]), (15,15,15)), axis=2), cmap=cmap)

    plt.axis('off')


    plt.subplot2grid((5,12), (4,9))
    plt.imshow(np.sum(np.reshape(U[:,1]/np.amax(U[:,1]), (15,15,15)),axis=2), cmap=cmap)
    plt.axis('off')
    plt.subplot2grid((5,12), (4,10))

    plt.imshow(np.sum(np.reshape(U[:,2]/np.amax(U[:,2]), (15,15,15)), axis=2), cmap=cmap)
    plt.axis('off')
    
    plt.subplot2grid((5,12), (4,11))
    plt.imshow(np.sum(np.reshape(U[:,3]/np.amax(U[:,3]), (15,15,15)), axis=2), cmap=cmap)
    plt.axis('off')

    #fig1.tight_layout()
    
    plt.show(block=True)
    return



if __name__ == "__main__":
    

    main()
