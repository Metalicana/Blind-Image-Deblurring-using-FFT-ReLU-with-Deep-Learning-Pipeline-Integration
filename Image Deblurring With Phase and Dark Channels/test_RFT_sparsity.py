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
    list_a = torch.zeros((1,640)).type(torch.float32)
    list_b = torch.zeros((1,640)).type(torch.float32)
    l = 0
    for i in range(80):
        for j in range(8):
            image_path = f'../../Levin_blurry/input80imgs8kernels/{i+1}_{j+1}_blurred.png'
            gt_path = f'../../Libin/trueture/img{i+1}_groundtruth_img.png'
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
            a[a<0.15]=0
            a = torch.nonzero(a)
            list_a[0,l] = int(a.size(0))
            # print(list_a[0,l].item())
            
            b = fft_relu(x)
            # b = b/torch.max(b)
            b = (b - b.min())/ (b.max() - b.min( ))
            b[b<0.15]=0
            b = torch.nonzero(b)
            
            list_b[0,l] = int(b.size(0))
            # print(b.item())
            l+=1

            # print(list_a[0,(i+1)*(j+1)-1], list_b[0,(i+1)*(j+1)-1])
    c = np.arange(1,641)
    # list_a/=1000.0
    # list_b/=1000.0
    plt.plot(c, list_a.squeeze().numpy(), label='RFT(B)', color='blue')

    # Plotting the second line graph with a different color
    plt.plot(c, list_b.squeeze().numpy(), label='RFT(I)', color='green')

    # Adding labels and title
    plt.xlabel('Image Index')
    plt.ylabel('L\u2080 Norm')
    plt.title('L\u2080 Norm of RFT(I) and RFT(B)')

    # Adding a legend
    plt.legend()

    # Displaying the graph
    plt.show()

    # print(list_a)
    # print(list_b)
    # bar_width = 0.35
    # print(list_b.squeeze().shape)
    # plt.figure(figsize=(8, 6))  # Optional: Adjust the figure size
    # plt.bar(c,list_a.squeeze().numpy(), label='RFT(B)')
    # plt.bar(c,list_b.squeeze().numpy(), label='RFT(I)')
    # plt.xlabel('Index')
    # plt.ylabel('L0Norm')
    # # plt.xticks(c + bar_width / 2, c)
    # plt.title('Bar Chart of RFT(I) and RFT(B)')
    # plt.legend()
    # plt.tight_layout()
    # plt.grid(True)
    # plt.show()  
            

if __name__ == "__main__":
    main()