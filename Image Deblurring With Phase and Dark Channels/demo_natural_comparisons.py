import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import blind_deconv
import ringing_artifacts_removal
# Import your Python implementations of necessary functions here.

# Define your blind_deconv function and other required functions here.

def main():
    # Specify your input image file path
    image_path = 'image/IMG_4355_small.png'

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
    }

    lambda_pixel = 4e-3
    lambda_grad = 4e-3
    lambda_tv = 0.002
    lambda_l0 = 2e-4
    weight_ring = 1

    is_select = False  # Set to True if you want to select a specific area for deblurring

    if is_select:
        # Allow the user to select a specific area for deblurring (not implemented in this example)
        pass
    else:
        yg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0

    # Perform blind deconvolution
    kernel, interim_latent = blind_deconv(yg, lambda_pixel, lambda_grad, opts)

    # Perform non-blind deconvolution
    saturation = 0  # Set this to 1 if the image is saturated
    if not saturation:
        # Apply TV-L2 denoising method
        Latent = ringing_artifacts_removal(yg, kernel, lambda_tv, lambda_l0, weight_ring)
    else:
        # Apply Whyte's deconvolution method
        Latent = whyte_deconv(yg, kernel)

    # Display the results
    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.imshow(kernel, cmap='gray')
    plt.title('Estimated Kernel')
    plt.subplot(132)
    plt.imshow(interim_latent, cmap='gray')
    plt.title('Interim Latent Image')
    plt.subplot(133)
    plt.imshow(Latent, cmap='gray')
    plt.title('Deblurred Image')

    # Save the results
    kernel_image = Image.fromarray((kernel * 255).astype('uint8'))
    latent_image = Image.fromarray((Latent * 255).astype('uint8'))
    interim_image = Image.fromarray((interim_latent * 255).astype('uint8'))

    kernel_image.save(os.path.join(results_dir, 'kernel.png'))
    latent_image.save(os.path.join(results_dir, 'result.png'))
    interim_image.save(os.path.join(results_dir, 'interim_result.png'))

if __name__ == "__main__":
    main()
