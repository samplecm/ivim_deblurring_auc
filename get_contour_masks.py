from PIL import Image, ImageDraw
from shapely.geometry.polygon import Polygon
import numpy as np
import copy
from copy import deepcopy
import scipy
import matplotlib.pyplot as plt
import mask_interpolation
import os
import pickle
from skimage import morphology, measure
from matplotlib.widgets import Slider, Button
from scipy.ndimage import binary_dilation, binary_erosion, convolve
import bisect
from utils import *

def get_max_percent_thres(img, percent=0.3, plot=False): 

 
    pet_voxels = img.ravel()
    #filter out non zero
    non_zero_mask = pet_voxels > 3
    pet_voxels = pet_voxels[non_zero_mask]
    pet_voxels_log = np.log(pet_voxels)
    bw_hist, bw_bins = np.histogram(pet_voxels_log, bins=150, range=(np.amin(pet_voxels_log), np.amax(pet_voxels_log)))


    #smooth the hist 

    bw_hist = scipy.signal.savgol_filter(bw_hist, 51,3)
    bw_hist = np.convolve(np.ones((11))/11, np.pad(bw_hist, (5,5), 'constant', constant_values=(0,0)))[15:-15] #remove 5 from each side extra to avoid analyzing edge region
    bw_bins = bw_bins[5:-5]
    #get largest gradient
    max_val = -10
    max_idx = None
    for idx in range(int(len(bw_hist)/6),int(len(bw_hist)/1.05)):
        val = bw_hist[idx]
        if val > max_val:
            max_idx = idx+1
            max_val = val
    if plot == True:
        fig, ax = plt.subplots()
        ax.plot(bw_bins[1:], bw_hist)
        plt.show(block=True)   



    #now check to left of max to see when it falls to percent of max
    # for idx in reversed(range(1,max_idx)):
    #     if bw_hist[idx] < percent * max_val: 
    #         final_idx = idx
    #         break
    return np.exp(bw_bins[max_idx]) * percent#np.exp(bw_bins[final_idx])

def get_pet_defined_masks(data_folder, plot=False):
    #function for updating slider plot of masks 
    def update(val):
        slice_index = int(slider.val)
        ax.imshow(psma_mask[slice_index, :,:])
        fig.canvas.draw_idle()

    def update2(val): #for label plot
        slice_index = int(slider.val)
        ax.imshow(label_image[slice_index, :,:])
        fig.canvas.draw_idle()    

    patient_nums = os.listdir(data_folder)

    patient_nums.sort()
    for patient_num in patient_nums:    
            
        img_series_path = os.path.join(os.getcwd(), "data", patient_num, "img_dict")
        #load image series
        with open(img_series_path, "rb") as fp:
            img_dict = pickle.load(fp)
        #load mask dictionary    
        with open(os.path.join(os.getcwd(), "data", patient_num, "mask_dict"), "rb") as fp:    
            mask_dict = pickle.load(fp)
        suv_factors = mask_dict["suv_factors"]

        img_orig = img_dict["PET"].image_array * suv_factors[0]

  
        
        #iterate over left and right glands
        for structure in mask_dict["PET"]:   #just indexing pet to get structures here

            if structure.lower() == "body":  #already pet defined
                continue

            ct_mask = mask_dict["PET"][structure].whole_roi_masks
            #suv_max = np.amax(ct_mask * img_orig)
            threshold_image = np.zeros_like(img_orig)
            threshold = get_max_percent_thres(img_orig * ct_mask, plot=False)
            #threshold = 0.35 * suv_max
            threshold_image[img_orig > threshold] = 1

            #now want to remove extending parotid/submandibulars from the current roi
            for other_structure in mask_dict["PET"]:
                if structure == other_structure or "Body" == other_structure:
                    continue
                threshold_image[binary_dilation(mask_dict["PET"][other_structure].whole_roi_masks, structure=np.ones((1,1,1)))] = 0    #remove the other region from the current so it doesnt leak over and extend through
            #remove islands
            threshold_image = binary_erosion(threshold_image, structure=np.ones((2,2,2)))
            threshold_image = binary_dilation(threshold_image, structure=np.ones((2,2,2)))
            label_image = measure.label(threshold_image)
            

            #now keep the mask that corresponds to the given structure
            intersec = binary_dilation(ct_mask, structure=np.ones((2,2,2))) * label_image 


            label_counts = np.bincount(np.ravel(intersec))
            label = np.argmax(label_counts[1:]) + 1
            psma_mask = (label_image == label)
            mask_indices = np.nonzero(psma_mask)
            min_slice = mask_indices[0].min()
            max_slice = mask_indices[0].max()

            #save the new pet mask over the ct mask
            mask_dict["PET"][structure].whole_roi_masks = psma_mask


            # fig, ax = plt.subplots()
            # ax.imshow(label_image[min_slice, :,:])
            # ax.set_title(f"{structure}")
            # ax_slider = plt.axes([0.2,0.01,0.65,0.03], facecolor='green')
            # slider = Slider(ax=ax_slider, label="Slice", valmin=0, valmax=max_slice+10, valstep=1, valinit=min_slice)
            # slider.on_changed(update2)
            # plt.show(block=True)
            #plot the new mask
            if plot == True:
                fig, ax = plt.subplots()
                ax.imshow(psma_mask[min_slice, :,:])
                ax.set_title(f"{structure}")
                ax_slider = plt.axes([0.2,0.01,0.65,0.03], facecolor='green')
                slider = Slider(ax=ax_slider, label="Slice", valmin=min_slice, valmax=max_slice, valstep=1, valinit=min_slice)
                slider.on_changed(update)
                plt.show(block=True)
        #save the mask dict for psma pet
        with open(os.path.join(os.getcwd(), "data", patient_num, "mask_dict_pet"), "wb") as fp:    
            pickle.dump(mask_dict, fp)
                


def get_all_structure_masks(img_series, structures_dict, segmentation_types=["reg", "si","ap","ml"], filled=True, img_type="reg"):
    if img_type == "reg":
        coords_array = img_series.coords_array_img
    elif img_type == "deblur":
        coords_array = img_series.deconv_coords_2x_img
    for structure_name in structures_dict:

        contours = structures_dict[structure_name].whole_roi_img
  
        mask_array = get_contour_masks(contours, coords_array, filled=filled)
        if img_type == "reg":
            structures_dict[structure_name].whole_roi_masks = mask_array
        elif img_type == "deblur":
            structures_dict[structure_name].whole_roi_masks_deblur = mask_array    
        for segmentation_type in segmentation_types:
            if img_type == "reg":
                if not hasattr(structures_dict[structure_name], "segmented_contours_" + segmentation_type):
                    continue
                #now get subsegment masks
                contours = getattr(structures_dict[structure_name], "segmented_contours_" + segmentation_type)
                subseg_masks = []
                for subseg in contours:
                    mask_array = get_contour_masks(subseg, coords_array)
                    subseg_masks.append(mask_array)
                setattr(structures_dict[structure_name], f"subseg_masks_{segmentation_type}", deepcopy(subseg_masks))   
                #structures_dict[structure_name].subseg_masks = deepcopy(subseg_masks)    
            elif img_type == "deblur":
                if not hasattr(structures_dict[structure_name], "segmented_contours_" + segmentation_type):
                    continue
                #now get subsegment masks
                contours = getattr(structures_dict[structure_name], "segmented_contours_" + segmentation_type)
                subseg_masks = []
                for subseg in contours:
                    mask_array = get_contour_masks(subseg, coords_array)
                    subseg_masks.append(mask_array)
                setattr(structures_dict[structure_name], f"subseg_masks_{segmentation_type}_deblur", deepcopy(subseg_masks))   
                #structures_dict[structure_name].subseg_masks = deepcopy(subseg_masks)       
    return structures_dict

def get_deconv_structure_masks(coords_array, structures_dict, segmentation_types=["si","ap","ml"]):




    for segmentation_type in segmentation_types:

        if not hasattr(structures_dict, "segmented_contours_" + segmentation_type):
            continue
        #now get subsegment masks
        contours = getattr(structures_dict, "segmented_contours_" + segmentation_type)
        subseg_masks = []
        for subseg in contours:
            mask_array = get_contour_masks(subseg, coords_array)
            subseg_masks.append(mask_array)
            #plot_3d_image(mask_array)
            setattr(structures_dict, f"subseg_masks_deconv_{segmentation_type}", deepcopy(subseg_masks))   
            #structures_dict[structure_name].subseg_masks = deepcopy(subseg_masks)       

def get_contour_masks(contours, array, filled=True):
    
    num_slices, len_y, len_x = array.shape[1:]
    contour_masks = np.zeros_like(array[0,:])
    contours = cartesian_to_pixel_coordinates(clone_list(contours), array)
    if filled == True:
        fill = 1
    else:
        fill = 0 
    #first get a list of z values with slices on them, and list of those slice images.
    mask_list = []
    z_list = []

    
    for contour in contours:
        contour_mask_filled = Image.new('L', (len_x, len_y), 0)        
        if contour == []:
            continue
        for slice in contour:
            if len(slice) < 3:
                continue
            contourPoints = []
            for point in slice:
                contourPoints.append((int(point[0]), int(point[1]))) #changed
            ImageDraw.Draw(contour_mask_filled).polygon(contourPoints, outline= 1, fill = fill)   
            mask_list.append(np.array(contour_mask_filled).astype(np.float32))   
            z_list.append(slice[0][2])
            break
    #now go through image slices and interpolate mask slices 
    for idx in range(num_slices):
        img_z = array[2,idx, 0,0]
        closest_slices = find_closest_slices(z_list, img_z, 0.5)
        if closest_slices is None:
            continue #slice is 0
        if type(closest_slices) == int:
            contour_masks[idx, :,:] = mask_list[closest_slices]
        elif type(closest_slices) == tuple:
            #need to interpolate between slices
            slice_1 = mask_list[closest_slices[0]]
            slice_2 = mask_list[closest_slices[1]]
            weight_1 = 1 - (img_z - z_list[closest_slices[0]])  /  (z_list[closest_slices[1]]- z_list[closest_slices[0]])
            weight_2 = 1 - weight_1
            interp_slice = slice_1 * weight_1 + slice_2 * weight_2
            #plot_2d_image(interp_slice)
            interp_slice = convolve(interp_slice, np.ones((2,2))/4)
            #plot_2d_image(interp_slice)
            interp_slice = interp_slice > 0.5
            contour_masks[idx, :, :] = interp_slice.astype(int)   
            # plot_2d_image(slice_1)
            # plot_2d_image(slice_2)
            # plot_2d_image(interp_slice)  
            n1= np.count_nonzero(slice_1)
            n2 = np.count_nonzero(slice_2)
            n3 = np.count_nonzero(interp_slice)

    return contour_masks.astype(np.bool)

def find_closest_slices(sorted_list, target_value, range_value):
    index = bisect.bisect_left(sorted_list, target_value)
    
    if 0 <= index < len(sorted_list):
        closest_value = sorted_list[index]
        if abs(closest_value - target_value) <= range_value:
            return index
        if index > 0:  
            closest_value = sorted_list[index-1]
            if abs(closest_value - target_value) <= range_value:
                return index
            
    if 0 < index < len(sorted_list):
        return index-1, index
    
    return None

# def get_contour_masks(contours, array, filled=True):
#     num_slices, len1, len2 = array.shape[1:]
#     contour_masks = np.zeros((num_slices,len1,len2), dtype=np.float32) 
#     contours = cartesian_to_pixel_coordinates(clone_list(contours), array)
#     if filled == True:
#         fill = 1
#     else:
#         fill = 0    
#     for idx in range(num_slices):#loop through all slices creating a mask for the contours
#         contour_mask_filled = Image.new('L', (len2, len1), 0)
#         slice_found = False
        
#         for contour in contours:
#             if contour == []:
#                 continue
#             for slice in contour:
#                 if len(slice) < 3:
#                     continue
#                 if abs(int(round(slice[0][2], 2)*100) - int(round(array[2,idx,0,0], 2)*100)) < 10: #if contour is on the current slice and only want to make new image when starting with first island
#                     slice_found = True
#                     contourPoints = []
#                     for point in slice:
#                         contourPoints.append((int(point[0]), int(point[1]))) #changed
#                     ImageDraw.Draw(contour_mask_filled).polygon(contourPoints, outline= 1, fill = fill)         

#                     # plt.imshow(contour_mask_filled)
#                     # plt.show()
#                     # unshown=False
#                     # print("")
#             if slice_found == True:
#                 break        
 
#         contour_masks[idx, :,:] = np.array(contour_mask_filled)       
#     #lastly, want to interpolate any missing slices. 
#     for idx in range(1,contour_masks.shape[0]-1):
#         if np.amax(contour_masks[idx,:,:])==0 and np.amax(contour_masks[idx-1,:,:])==1 and np.amax(contour_masks[idx+1,:,:]) == 1:
#             interp_img = mask_interpolation.interp_shape(contour_masks[idx-1,:,:], contour_masks[idx+1,:,:], 0.5)
#             contour_masks[idx,:,:] = interp_img
#             # fig,ax = plt.subplots(3)
#             # ax[0].imshow(contour_masks[idx-1,:,:])
#             # ax[1].imshow(contour_masks[idx,:,:])
#             # ax[2].imshow(contour_masks[idx+1,:,:])
#             # plt.show()
#             # print("")
#     return contour_masks.astype(np.bool)             

def clone_list(list):
    listCopy = copy.deepcopy(list)
    return listCopy  

def cartesian_to_pixel_coordinates(contours, array):
    #convert x and y values for a contour into the pixel indices where they are on the pet array
    xVals = array[0,0,0,:]
    yVals = array[1,0,:,0]
    for contour in contours: 
        if len(contour) == 0:
            continue
        for slice in contour:
            if len(slice) == 0: continue
            for point in slice:
                point[0] = min(range(len(xVals)), key=lambda i: abs(xVals[i]-point[0]))
                point[1] = min(range(len(yVals)), key=lambda i: abs(yVals[i]-point[1]))
    return contours  
