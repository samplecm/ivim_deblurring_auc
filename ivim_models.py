import numpy as np
import copy
import pickle
#import image_processing
from astropy.io import fits

import os
from utils import plot_3d_image
from scipy.optimize import curve_fit, least_squares

def get_ivim_model_images(data_path, patient_nums, load_types):
    mean_params_list = []
    for patient_num in patient_nums:
    #do analysis for both pre and post RT
        patient = "SIVIM" + patient_num  
        for scan_type in load_types[patient]:
            
            print(f"Getting ivim model images for {patient} {scan_type}-RT")
            with open(os.path.join(os.getcwd(), "mri_processed_data", str(patient + scan_type + "diffusion_imgs")), "rb") as fp:
                img_dict = pickle.load(fp)     

            #Visuals.plot_signal_vs_b([[data[0], data[2]] for data in all_imgs_ss], use_b0=False)         
            if load_types[patient][scan_type] != "adc":
                imgs = img_dict["diffusion"].image_arrays #dictionary of image with b values
                # for b in imgs:
                #     Visuals.plot_biexp(imgs[b])
                adc_img = get_adc_image(imgs)
                plot_3d_image(adc_img)
                with open(os.path.join(os.getcwd(), "mri_processed_data", str(patient + scan_type + "adc_img")), "wb") as fp:
                    pickle.dump(adc_img, fp)
                # d_img, f_img, p_img = get_biexp_image(imgs, use_b0=False)
                # with open(os.path.join(os.getcwd(), "mri_processed_data", str(patient + scan_type + "d_img")), "wb") as fp:
                #     pickle.dump(d_img, fp)
                # with open(os.path.join(os.getcwd(), "mri_processed_data", str(patient + scan_type + "f_img")), "wb") as fp:
                #     pickle.dump(f_img, fp)
                # with open(os.path.join(os.getcwd(), "mri_processed_data", str(patient + scan_type + "p_img")), "wb") as fp:
                #     pickle.dump(p_img, fp)      
                
            else: #for sivim08, post scan 
                d_img = img_dict["diffusion"].image_array   
                with open(os.path.join(os.getcwd(), "mri_processed_data", str(patient + scan_type + "d_img")), "wb") as fp:
                    pickle.dump(d_img, fp)  
            #export_biexp_fits(biexp_img, fits_data, patient, scan_type)

            # Visuals.plot_biexp(d_img)
            # Visuals.plot_biexp(f_img)
            # Visuals.plot_biexp(p_img)


    print("Finished")



def get_adc_image(image_data):
    #image_data is a list with items that are 2 element lists.
    #1st element = image list at b value
    #2nd element = b value

    #excluding b = 0 --> S(b) = S(b_0)exp(-(b-b_0)ADC)
    #so the fitting procedure is the same as always, with the small difference that b --> b-b_0

    #using least squares for ADC:

    #first get size of image and make a blank one of same size for image.
    adc_img_list = []    #list of images at every slice
    b_values = sorted(list(image_data.keys()))
    b_0 = b_values[0]   #the lowest b value in the lot
    img_size = image_data[b_0][0].shape
    from utils import plot_3d_image
    plot_3d_image(image_data[b_0])
    #loop through all image slices
    for slice_idx in range(len(image_data[b_0])):
        #we need a temporary array for the numerator sum and one for the denominator sum
        adc_num = np.zeros(img_size)
        adc_den = np.zeros(img_size)
        #now go through the images at different b values and add values to the adc sum (defined for least squares, see latex document 5.1)
        
        image_b0 = image_data[b_0][slice_idx]
        
        

        for b in b_values[1:]:    #start at 1 because b_0 image not useful for fitting
            image = image_data[b][slice_idx]
            adc_num -= (b-b_0) * np.log(image / image_b0) 
            adc_den += (b-b_0)**2

        adc_img = adc_num / adc_den
        adc_img[adc_img == np.inf] = np.nan
        adc_img_list.append(adc_img)
        # for row_idx in range(img_size[0]):
        #     for column_idx in range(img_size[1]):
        #         continue
    adc_img = np.zeros((len(adc_img_list), adc_img_list[0].shape[0],adc_img_list[0].shape[1]))
    for idx in range(len(adc_img_list)):
        adc_img[idx, :,:] = adc_img_list[idx]

    return adc_img
def get_biexp_image(image_data, segmented=True, use_b0=False):

    #nested function for fitting:
    def biexp_function(b_array, f, pseudo_d):    #biexponential function calculated without the b=0 signal image
        val = f * np.exp(-b_array * pseudo_d) + (1-f) * np.exp(-b_array * d)    
        return val

    image_data = copy.deepcopy(image_data)
    #excluding b = 0 --> S(b) = S(b_0)exp(-(b-b_0)ADC)
    #so the fitting procedure is the same as always, with the small difference that b --> b-b_0

    #using least squares for ADC:

    #first get size of image and make a blank one of same size for image.
    b_values = list(image_data.keys())
    b0 = b_values[0] 

    num_slices, row_size, column_size = image_data[b0].shape
    f_img = np.zeros_like(image_data[b0])
    d_img = np.zeros_like(image_data[b0])
    p_img = np.zeros_like(image_data[b0])


    #first collect all images with b-b_0 > 200 (b0 is smallest b value) and use these to first fit the diffusion coefficient
    high_b_list = []
    if use_b0:
        fitting_b_list = np.array(b_values)
    elif not use_b0:
        fitting_b_list = np.array(b_values[1:])    

    for b in b_values:
        if b - b0 >= 200:
            high_b_list.append(b)      

    if b0 > 0:    
        b_values = [b-b0 for b in b_values]

    #Normalize images by signal at b0

    for b in reversed(b_values):
        image_data[b] /= image_data[b0]

    for slice_idx in range(num_slices):
        #see the latex document of IVIM fitting for the formula used here for fitting. 
        #first need to get the average b-b0 nd y                
        for r in range(row_size):
            for c in range(column_size):
                #get normalized signal values in array
                #now use NLLS for D*, and f (using f calculated before as initial guess)
                
                signals = []
                high_signals = []

                for b in fitting_b_list:    #get all normalized signals into a list (not using b0)
                    signals.append(image_data[b][slice_idx][r,c])
                    if b in high_b_list:
                        high_signals.append(image_data[b][slice_idx][r,c])

                    
                if np.isnan(signals).any() or np.isinf(signals).any():
                    d_img[slice_idx ,r,c] = np.nan
                    f_img[slice_idx,r,c] = np.nan
                    p_img[slice_idx, r,c] = np.nan
                    continue


                signals = np.array(signals)
                sum_xy = 0
                sum_x = 0
                sum_y = 0
                sum_x2 = 0
                sum_x_2 = 0
                N = len(high_b_list)
                for b_idx, b in enumerate(high_b_list):
                    y = np.log(high_signals[b_idx])
                    sum_y += y
                    sum_x += b
                    sum_xy += y * b
                    sum_x2 += b**2
                sum_x_2 = sum_x ** 2
                M = (N * sum_xy - (sum_x * sum_y)) / (N * sum_x2 - sum_x_2)
                B = (sum_y - M * sum_x ) / N

                d = max(-M, 0)
                f = 1 - np.exp(B)

                p0 = [max(f, 0), 0.04]   #initial parameters for fitting f and p
                #D_voxel = D[r,c]
                try:
                    popt, pcov = curve_fit(biexp_function, fitting_b_list, signals, p0=p0, method='trf', bounds=([0, 0.004],[1,0.5]))
                    f, pseudo_d = popt
                    f_img[slice_idx,r,c,] = f
                    p_img[slice_idx,r,c,] = pseudo_d
                    d_img[slice_idx,r,c,] = d

                except: #curve fit doesnt converge
                    f_img[slice_idx,r,c,] = np.nan
                    p_img[slice_idx,r,c,] = np.nan
                    d_img[slice_idx,r,c,] = d      

    return d_img, f_img, p_img 




def export_biexp_fits(model_list, fits_data, patient, scan_type):
    #first need to replace the fits_data images with the new model images. slices are in same order.
    #  
    f_data = copy.deepcopy(fits_data)
    d_data = copy.deepcopy(fits_data)
    pd_data = copy.deepcopy(fits_data)

    for idx, _ in enumerate(fits_data):
        f_data[idx].data = np.flip(np.around(model_list[idx][0][:,:,0]*1000, decimals=4), axis=0)
        pd_data[idx].data = np.flip(np.around(model_list[idx][0][:,:,1]*1000, decimals=4), axis=0)
        d_data[idx].data = np.flip(np.around(model_list[idx][0][:,:,2]*1000, decimals=4), axis=0)

        #convert to float32
        d_data[idx].data = d_data[idx].data.astype('float32')
        pd_data[idx].data = pd_data[idx].data.astype('float32')
        f_data[idx].data = f_data[idx].data.astype('float32')
        

    export_dir = os.path.join(os.getcwd(), "exports")
    f_data.writeto(os.path.join(export_dir, str(patient + "_" + scan_type + "_f.fits")), overwrite=True)
    d_data.writeto(os.path.join(export_dir, str(patient + "_" + scan_type + "_d.fits")), overwrite=True)
    pd_data.writeto(os.path.join(export_dir, str(patient + "_" + scan_type + "_pd.fits")), overwrite=True)

    return 
