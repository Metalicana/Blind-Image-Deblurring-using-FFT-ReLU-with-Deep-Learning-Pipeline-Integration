import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from blind_deconv import blind_deconv
from ringing_artifacts_removal import ringing_artifacts_removal
from misc import visualize_rgb ,visualize_image, gray_image, process_image,PSNR
from metrics import psnr, ssim
from skimage.metrics import structural_similarity as s_sim
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
    for j in range(4):
        x = 0
        for i in range(8):
            
            # real_path = f'../../saturate_100_for_test/testset_public/gt_gray/{i+1}.png'
            real_path = f'groundtruths/Levin/im0{j+1}_ker0{i+1}.png'
            # real_path = f'groundtruths/Levin/1_1.png'
            # image_path = f'../../Levin_sharp/groundtruth_kernel_latent_zoran/Kernel_{i+1}/{j+1}_gtk_latent_zoran.png'
            # deblurred_path = f'results/result.png'
            deblurred_path = f'results/Levin_dynamic/Levin_Radi_{j+1}_{i+1}_best.png'
            # deblurred_path = f'../../cvpr16_deblurring_code_v1/results/phase/{j+1}_{i+1}_result.png'
            # deblurred_path = f'results/Levin_dynamic/pan/im0{j+5}_ker0{i+1}.png'
            real = gray_image(Image.open(real_path))
            radi = gray_image(Image.open(deblurred_path))
            if(real.shape != radi.shape):
                print(0)
                continue
            # print(ssim(real_path, deblurred_path))
            # print(s_sim(real,radi[50:-50,50:-50] ))
            from misc import PSNR
            # l += ssim(real_path, deblurred_path)
            # x += PSNR(real,radi )
            # print(ssim(real_path, deblurred_path))
            print(PSNR(real,radi ))

            # print(PSNR(real,radi ))
            # x += PSNR(real,radi )
            # print(PSNR(real,radi ))
            # print(ssim(real_path,deblurred_path ))
        # print(x/8.0)
        # print(l/8.0)
            # print(img[25:-25,25:-25].shape)

        
    
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
