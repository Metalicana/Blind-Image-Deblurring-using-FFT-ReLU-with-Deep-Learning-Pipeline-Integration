import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from blind_deconv import blind_deconv
from ringing_artifacts_removal import ringing_artifacts_removal
from misc import visualize_rgb ,visualize_image, gray_image, process_image,PSNR, fft_relu, findM
from metrics import psnr
import numpy as np
def main():
    list_a = torch.zeros((1,48)).type(torch.float32)
    list_b = torch.zeros((1,48)).type(torch.float32)
    l = 0
    for i in range(4):
        for j in range(12):
            image_path = f'images/blurry{i+1}_{j+1}.png'
            gt_path = f'groundtruths/GroundTruth{i+1}_{j+1}_1.png'
            results_dir = 'results'

            os.makedirs(results_dir, exist_ok=True)
            x = process_image(Image.open(gt_path))
            x = x.permute(1,2,0)
            x = x/255.0
            y = process_image(Image.open(image_path))
            y = y.permute(1,2,0)
            y = y/255.0
            # a = findM(y)
            a = fft_relu(y)
            a = (a - a.min())/ (a.max() - a.min( ))
            a = torch.sum(a)
            list_a[0,l] = a
            # print(list_a[0,l].item())
            
            b = fft_relu(x)
            # b = b/torch.max(b)
            b = (b - b.min())/ (b.max() - b.min( ))
            b = torch.sum(b)
            list_b[0,l] = b
            print(b.item())
            l+=1

            # print(list_a[0,(i+1)*(j+1)-1], list_b[0,(i+1)*(j+1)-1])
    c = np.arange(1,49)
    # list_a/=1000.0
    # list_b/=1000.0
    print(list_a)
    print(list_b)
    bar_width = 0.35
    print(list_b.squeeze().shape)
    plt.figure(figsize=(8, 6))  # Optional: Adjust the figure size
    plt.bar(c,list_a.squeeze().numpy(), label='RFT(B)')
    plt.bar(c,list_b.squeeze().numpy(), label='RFT(I)')
    plt.xlabel('Index')
    plt.ylabel('L0Norm')
    # plt.xticks(c + bar_width / 2, c)
    plt.title('Bar Chart of RFT(I) and RFT(B)')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()  
            

if __name__ == "__main__":
    main()