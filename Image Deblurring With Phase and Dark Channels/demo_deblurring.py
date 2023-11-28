import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from blind_deconv import blind_deconv
from ringing_artifacts_removal import ringing_artifacts_removal
from misc import visualize_rgb ,visualize_image, gray_image, process_image
# Import your Python implementations of necessary functions here.

# Define your blind_deconv function and other required functions here.


def main():
    # Specify your input image file path
    image_path = 'images/plate.png'

    # Create the results directory if it doesn't exist
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Load the image
    image = cv2.imread(image_path)

    # Set parameters
    opts = {
        'prescale': 1,   # Downsampling
        'xk_iter': 5,    # Iterations
        'gamma_correct': 1.0,
        'k_thresh': 20,
        'kernel_size':25,
    }

    lambda_dark = 4e-3
    #Experimenting with lambda_dark set to 0
    lambda_dark = 0
    lambda_grad = 4e-3
    lambda_tv = 0.003
    lambda_l0 = 5e-4
    weight_ring = 1
    is_select = False  # Set to True if you want to select a specific area for deblurring

    if is_select:
        # Allow the user to select a specific area for deblurring (not implemented in this example)
        pass
    else:
        inpt = Image.open(image_path)
        yg = gray_image(inpt)

        # print(yg[0:5,0:5])
    # Perform blind deconvolution
    
    kernel, interim_latent = blind_deconv(yg, lambda_dark, lambda_grad, opts)
    print('kernel below')
    print(kernel.detach())
    plt.figure(figsize=(12, 6))
    plt.imshow(kernel.detach(), cmap='gray')
    plt.title('Estimated Kernel')
    plt.show()
    # Perform non-blind deconvolution
    saturation = 0  # Set this to 1 if the image is saturated
    if not saturation:
        # Apply TV-L2 denoising method
        y = process_image(Image.open(image_path))
        y = y.permute(1,2,0)
        # print(y.shape)
        Latent = ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring)
    else:
        # Apply Whyte's deconvolution method
        # Latent = whyte_deconv(yg, kernel)
        pass
    # print(Latent.shape)
   
    Latent = Latent/255.0
    visualize_rgb(Latent)
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
