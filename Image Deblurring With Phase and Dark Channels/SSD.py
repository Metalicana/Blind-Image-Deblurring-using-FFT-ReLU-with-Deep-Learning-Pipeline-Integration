import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from blind_deconv import blind_deconv
from ringing_artifacts_removal import ringing_artifacts_removal
from misc import visualize_rgb ,visualize_image, gray_image, process_image,PSNR
from metrics import psnr
# Import your Python implementations of necessary functions here.

# Define your blind_deconv function and other required functions here.


def main():
    # Specify your input image file path
   
    
    # list = [25, 31, 35, 39, 41, 45, 49]
    # list = [ 4e-5, 3e-5, 2e-5 ,4e-3, 3e-3, 2e-3,4e-2, 3e-2, 2e-2]
    # list = [4.2e-2,4.22e-2,4.18e-2,4.3e-2,3.8e-2]
    #4.2e-2, 4e-2, 4e-3, 4.1e-2
    # real_path = f'../../Levin_sharp/trueture/img{j+1}_groundtruth_img.png'
    # image_path = f'../../Levin_sharp/groundtruth_kernel_latent_zoran/Kernel_{i+1}/{j+1}_gtk_latent_zoran.png'
    # deblurred_path = f'results/Levin_Radi_{j+1}_{i+1}.png'
    l = 0
    x = 0
    list = []
    for j in range(80):
        for i in range(8):
            real_path = f'../../Levin_sharp/trueture/img{j+1}_groundtruth_img.png'
            image_path = f'../../Levin_sharp/groundtruth_kernel_latent_zoran/Kernel_{i+1}/{j+1}_gtk_latent_zoran.png'
            deblurred_path = f'results/Levin_Radi_{j+1}_{i+1}.png'
            # deblurred_path = f'../../Levin_sharp/all_deblur_results/img{j+1}_kernel{i+1}_ChoAndLee_img.png'

            real = Image.open(real_path)
            real = gray_image(real)

            img = Image.open(image_path)
            img = gray_image(img)
            # img /= 255.0
            deblur = Image.open(deblurred_path)
            # deblur /= 255.0?
            deblur = gray_image(deblur)
            # print(img[25:-25,25:-25].shape)
        
            diff1 = img[75:-75,75:-75] - real
            diff1 = diff1**2
            diff1 = torch.sum(diff1)
            diff2 = deblur - real

            # diff2 = deblur[50:-50,50:-50]- real
            diff2 = diff2**2
            diff2 = torch.sum(diff2)
            if l>100:
                break
            ans = (diff2 / diff1).item()
            print(ans)
            list.append(ans)
            if ans <5:
                x+=1
            l+=1
        if l > 100:
            break;
    list.sort()
    print(x)
    print(list)
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
