import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from blind_deconv import blind_deconv
from ringing_artifacts_removal import ringing_artifacts_removal
from misc import visualize_rgb ,visualize_image, gray_image, process_image,PSNR, average_surrounding
from metrics import psnr
# Import your Python implementations of necessary functions here.

from gt_kernels import im01_ker01_kernel, im01_ker02_kernel, im01_ker03_kernel, im01_ker04_kernel, im01_ker05_kernel, im01_ker06_kernel, im01_ker07_kernel, im01_ker08_kernel, im02_ker01_kernel, im02_ker02_kernel, im02_ker03_kernel, im02_ker04_kernel, im02_ker05_kernel, im02_ker06_kernel, im02_ker07_kernel, im02_ker08_kernel, im03_ker01_kernel, im03_ker02_kernel, im03_ker03_kernel, im03_ker04_kernel,im03_ker05_kernel, im03_ker06_kernel,im03_ker07_kernel, im03_ker08_kernel, im04_ker01_kernel, im04_ker02_kernel, im04_ker03_kernel, im04_ker04_kernel, im04_ker05_kernel, im04_ker06_kernel, im04_ker07_kernel, im04_ker08_kernel

# x = ker_list[0]
# print(x.shape)
# print(x.max())
# print(x.min())
# visualize_image(x)

def main():
    # Specify your input image file path
    
    #import all kernel tensors from gt_kernels.py
    ker_list = [im01_ker01_kernel, im01_ker02_kernel, im01_ker03_kernel, im01_ker04_kernel, im01_ker05_kernel, im01_ker06_kernel, im01_ker07_kernel, im01_ker08_kernel, im02_ker01_kernel, im02_ker02_kernel, im02_ker03_kernel, im02_ker04_kernel, im02_ker05_kernel, im02_ker06_kernel, im02_ker07_kernel, im02_ker08_kernel, im03_ker01_kernel, im03_ker02_kernel, im03_ker03_kernel, im03_ker04_kernel,im03_ker05_kernel, im03_ker06_kernel,im03_ker07_kernel, im03_ker08_kernel, im04_ker01_kernel, im04_ker02_kernel, im04_ker03_kernel, im04_ker04_kernel, im04_ker05_kernel, im04_ker06_kernel, im04_ker07_kernel, im04_ker08_kernel]
    
    
    # list = [25, 31, 35, 39, 41, 45, 49]
    # list = [ 4e-5, 3e-5, 2e-5 ,4e-3, 3e-3, 2e-3,4e-2, 3e-2, 2e-2]
    # list = [4.2e-2,4.22e-2,4.18e-2,4.3e-2,3.8e-2]
    # list = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3,8e-3,9e-3,1e-2]
 
    list = [15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 39, 41, 45, 49, 55, 43]
    idx_list = [8,5,7,9,13,13,7,10,9,5,5,15,5,9,9,10,5,6,14,9,5,4,9,10,5,7,5,7,4,6,14,13]
    #Open file for writing
    f = open("results.txt", "w")  
    #4.2e-2, 4e-2, 4e-3, 4.1e-2
    count = 0
    for j in range(4):
        for i in range(8):
            # if j == 1 and i == 3:
            #     print('hehe')
            # else:
            #     count+=1
            #     continue
            # for ker_size in range(15):
            #     print(ker_size, i,j)
            #     if j == 0:
            #         continue
            #     if j == 1 and i <= 4:
            #         continue 
            #     if ker_size == 0 and i == 7 and j == 1:
            #         continue 
            #     if ker_size == 1 and i == 7 and j == 1:
            #         continue
            #     if ker_size == 2 and i == 7 and j == 1:
            #         continue 
            image_path = f'images/Levin/im0{j+1}_ker0{i+1}.png'
            # print()
            # Create the results directory if it doesn't exist
            results_dir = f'results/Levin_dynamic'
            os.makedirs(results_dir, exist_ok=True)

            # Load the image
            image = cv2.imread(image_path)
        # Set parameters
            opts = {
                'prescale': 1,   # Downsampling
                'xk_iter': 5,    # Iterations
                'gamma_correct': 1.0,
                'k_thresh': 20,
                'kernel_size':list[idx_list[count]],
            }
            count += 1

            lambda_dark = 4e-3
            #Experimenting with lambda_dark set to 0
            lambda_ftr = 3.5e-4
            lambda_dark = 0
            lambda_grad = 4e-3


            lambda_tv = 0.001
            lambda_l0 = 1.5e-4
            # lambda_l0 = list[i]
            weight_ring = 1
            is_select = False  # Set to True if you want to select a specific area for deblurring

            if is_select:
                # Allow the user to select a specific area for deblurring (not implemented in this example)
                pass
            else:
                inpt = Image.open(image_path)
                
                yg = gray_image(inpt)
                # if i == 5:
                #     yg = yg[50:-50,50:-50]
                # yg = yg[50:-50,50:-50]

                # print(yg[0:5,0:5])
            # Perform blind deconvolution
            
            kernel, interim_latent = blind_deconv(yg, lambda_ftr,lambda_dark, lambda_grad, opts)
            # plt.figure(figsize=(12, 6))
            # plt.imshow(kernel, cmap='gray')
            # plt.title('Estimated Kernel')
            # plt.show()
            # Perform non-blind deconvolution
            saturation = 0  # Set this to 1 if the image is saturated
            if not saturation:
                # Apply TV-L2 denoising method
                
                from misc import process_gray
                y = process_gray(Image.open(image_path))
                print(f'Yshape {y.shape}')
                y = y.permute(1,2,0)
                # print(y.max())

                
                # gt_kernel_path = f'groundtruths/Levin/kernel{i+1}_groundtruth_kernel.png'
                # gt_kernel = process_image(Image.open(gt_kernel_path))
                # gt_kernel = gt_kernel.squeeze()
                # # gt_kernel = gt_kernel/255.0
                # gt_kernel = gt_kernel / gt_kernel.sum()
                # visualize_image(gt_kernel)
                # visualize_image(kernel)
                # y = y[50:-50,50:-50]
                # print(y[0:5,0:5,0])
                # print(y.shape)
                Latent = ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring)
                # ker_idx = (j*4) + i
                # x =torch.flip(ker_list[ker_idx], [0,1])
                gt_kernel_path = f'groundtruths/Levin/gt_kernels/im0{j+1}_ker0{i+1}.png'
                gt_kernel = process_gray(Image.open(gt_kernel_path))
                gt_kernel = gt_kernel.squeeze()
                gt_kernel = torch.flip(gt_kernel, [0,1])
                # gt_kernel = gt_kernel/255.0
                gt_Latent = ringing_artifacts_removal(y, gt_kernel, lambda_tv, lambda_l0, weight_ring)
                # print(gt_Latent.max())
                # print(gt_Latent.min())
                # print(Latent.max())
                # print(Latent.min())
                # visualize_image(gt_Latent)
                # visualize_image(Latent)
                # visualize_rgb(Latent)
                # print(gt_Latent.shape)
            else:
                # Apply Whyte's deconvolution method
                # Latent = whyte_deconv(yg, kernel)
                pass
            # print(Latent.shape)
            # print(Latent.max())
            # Latent = Latent/255.0
            # print(Latent[0:5,0:5,0])
            # visualize_rgb(Latent)
            #save the Latent matrix as a JPG image in the results folder
            
            #Calculate Square error with groundtruth
            
            #First load the groundtruth image
            gt_path = f'groundtruths/Levin/im0{j+1}_ker0{i+1}.png'
            gt = process_gray(Image.open(gt_path)).permute(1,2,0)
            # gt = gt / 255.0
            # print(gt.shape)
            # print(Latent.shape)
            #write to results.txt
            error_ratio = torch.sum((gt - Latent)**2) / torch.sum((gt - gt_Latent)**2)
            print(torch.sum((gt - Latent)**2))
            print(torch.sum((gt - gt_Latent)**2))
            f.write(f'Levin_Radi_{j+1}_{i+1}_best : {error_ratio}\n')

            # gt_Latent[gt_Latent>1.0] = 1.0
            # gt_Latent[gt_Latent<0.0] = 0.0
            # gt_Latent = gt_Latent*255.0
            # gt_Latent = gt_Latent.squeeze()
            # gt_Latent = gt_Latent.numpy()
            # gt_Latent = gt_Latent.astype('uint8')
            # gt_Latent = Image.fromarray(gt_Latent)
            # gt_Latent.save(os.path.join(results_dir, f'Levin_Radi_{j+1}_{i+1}_{ker_size}_gt.png'))


            Latent[Latent>1.0] = 1.0
            Latent[Latent<0.0] = 0.0
            Latent = Latent*255.0
            # Latent = average_surrounding(Latent)
            Latent = Latent.squeeze()
            Latent = Latent.numpy()
            Latent = Latent.astype('uint8')
            Latent = Image.fromarray(Latent)
            # Latent.save(os.path.join(results_dir, f'Levin_Radi_{j+1}_{i+1}_{ker_size}.png'))
            
            Latent.save(os.path.join(results_dir, f'Levin_Radi_{j+1}_{i+1}_best.png'))

            kmn = kernel.min()
            kmx = kernel.max()
            kernel = (kernel - kmn)/(kmx - kmn)
            kernel = kernel*255.0
            kernel = kernel.numpy()
            kernel = kernel.astype('uint8')
            kernel = Image.fromarray(kernel)
            # kernel.save(os.path.join(results_dir, f'Levin_Radi_{j+1}_{i+1}_{ker_size}_kernel.png'))
            
            kernel.save(os.path.join(results_dir, f'Levin_Radi_{j+1}_{i+1}_best_kernel.png'))
    #Close file
    f.close()
# Lmx = Latent.max()
    # Lmn = Latent.min()
    # Latent = (Latent - Lmn)/(Lmx - Lmn)
    # visualize_rgb(Latent)
    # Display the results
    # plt.figure(figsize=(12, 6))
    # plt.subplot(131)
    # plt.imshow(kernel, cmap='gray')
    # plt.title('Estimated Kernel')
    # plt.subplot(132)
    # plt.imshow(interim_latent, cmap='gray')
    # plt.title('Interim Latent Image')
    # plt.subplot(133)
    # plt.imshow(Latent, cmap='gray')
    # plt.title('Deblurred Image')

    # # Save the results
    # kernel_image = Image.fromarray((kernel * 255).astype('uint8'))
    # latent_image = Image.fromarray((Latent * 255).astype('uint8'))
    # interim_image = Image.fromarray((interim_latent * 255).astype('uint8'))

    # kernel_image.save(os.path.join(results_dir, 'kernel.png'))
    # latent_image.save(os.path.join(results_dir, 'result.png'))
    # interim_image.save(os.path.join(results_dir, 'interim_result.png'))

if __name__ == "__main__":
    main()
