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
import time
import sys
from misc import conv2
# Import your Python implementations of necessary functions here.

# Define your blind_deconv function and other required functions here.


def main():
    sharp_path = 'images/sharp.jpg'
    kernel_path = 'images/kernel.png'
    sharp = Image.open(sharp_path)
    kernel = Image.open(kernel_path)
    sharp = process_image(sharp)
    kernel = gray_image(kernel).squeeze()
    ans = conv2(sharp, kernel, shape='same')
    print(ans)

if __name__ == "__main__":
    main()
