## Update
We are officially accepted to "IEEE/CVF Winter Conference on Applications of Computer Vision" (WACV2025) 🔥🔥

## The First PyTorch implementation of popular blind image deblurring papers
We present this PyTorch implementation, shifting from MATLAB for more reachability to researchers. There are scopes to optimize speed, and integrate this into deep learning frameworks! We also have included our own prior into the blind image deblurring framework.

## Blind Image Deblurring
Blind image deblurring is the process of deriving a sharp image and a blur kernel from a blurred image. Blurry images are typically modeled as the convolution of a sharp image with a blur kernel, necessitating the estimation of the unknown blur kernel to perform blind image deblurring effectively. Existing approaches primarily focus on domain-specific features of images, such as salient edges, dark channels, and light streaks. These features serve as probabilistic priors to enhance the estimation of the blur kernel. For improved generality, we propose a novel prior (ReLU sparsity prior) that estimates blur kernel effectively across all distributions of images (natural, facial, text, low-light, saturated, etc). Our approach demonstrates superior efficiency, with inference times up to three times faster, while maintaining high accuracy in PSNR, SSIM, and error ratio metrics. We also observe a noticeable improvement in the performance of the state-of-the-art architectures (in terms of the aforementioned metrics) in deep learning-based approaches when our method is used as a post-processing unit.

## How to Run
1. Keep images in the images folder
2. run python3 demo_deblurring.py image_name.extension kernel_size

## Consider Citing Us
@misc{radi2024blindimagedeblurringusing,
      title={Blind Image Deblurring using FFT-ReLU with Deep Learning Pipeline Integration}, 
      author={Abdul Mohaimen Al Radi and Prothito Shovon Majumder and Syed Mumtahin Mahmud and Mahdi Mohd Hossain Noki and Md. Haider Ali and Md. Mosaddek Khan},
      year={2024},
      eprint={2406.08344},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.08344}, 
}
