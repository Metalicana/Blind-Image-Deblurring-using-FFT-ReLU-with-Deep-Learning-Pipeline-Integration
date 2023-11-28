import torch
from misc import fft2, ifft2
def fft_relu(input_tensor):
    fft_input_tensor = fft2(input_tensor)
    real_part = torch.real(fft_input_tensor)
    imag_part = torch.imag(fft_input_tensor)

    # Apply ReLU separately to real and imaginary parts
    real_part_relu = torch.relu(real_part)
    imag_part_relu = torch.relu(imag_part)

    # Combine real and imaginary parts back into a complex tensor
    fft_tensor_relu = torch.complex(real_part_relu, imag_part_relu)
    result = torch.real(ifft2(fft_tensor_relu)) - 0.5*input_tensor
    # result = (result - result.min())/(result.max() -result.min())
    return result