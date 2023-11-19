import torch
import torch.fft as fft
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
def gray_image(inpt):
    image = process_image(inpt)
    image = image.permute(1,2,0)
    yg =  image[:,:,0]*0.2989+ image[:,:,1]*0.587 + image[:,:,2]*0.114
    yg = torch.round(yg)
    yg = yg / 255.0
    return yg

def process_image(input_image) -> torch.Tensor:
  transform = transforms.Compose([
      transforms.PILToTensor()
  ])
  input_tensor = transform(input_image).type(torch.float32)
  return input_tensor

def visualize_image(input_tensor):
    plt.imshow(input_tensor ,cmap='gray')
    plt.title("Image")
    plt.colorbar()
    plt.show()
def visualize_rgb(input_tensor):
    plt.imshow(input_tensor.permute(1,2,0))
    plt.title("Image")
    plt.colorbar()
    plt.show()


def dst(x):
    N, M = x.shape
    xx = torch.arange(1,N+1).view(-1,1)
    yy = torch.arange(1,N+1)
    
    sinMat = torch.sin(xx * yy * torch.pi / (N+1))
    ans = torch.matmul(x.t(), sinMat).t()

    return ans

def idst(x):
    N, M = x.shape
    xx = torch.arange(1,N+1).view(-1,1)
    yy = torch.arange(1,N+1)
    
    sinMat = torch.sin(xx * yy * torch.pi / (N+1))
    ans = torch.matmul(x.t(), sinMat).t()
    ans = ans * 2 /(N+1)
    return ans

def conv2(A, B, shape):
    #Pad A such that A has dimension (A.shape[0] + 2*B.shape[0] -2, A.shape[1] + 2*B.shape[1] - 2)
    #You do regular convolution.
    #then if it is valid, return just the center crop version of it
    #Expect A and B to be 2 dimensional matrieces
    #if shape is full then return full
    if A.dim() == 2:
        # print(f'and finally A = {A.shape} B = {B.shape}')
        padCol = int((B.shape[1]*2 - 2)/2)
        padRow = int((B.shape[0]*2 - 2)/2)
        cropX, cropY = (A.shape[0]-B.shape[0]+1, A.shape[1]-B.shape[1]+1)
        
        padding = (padCol, padCol, padRow, padRow)
        A = F.pad(A,padding, value=0)
        
        B = torch.flip(B,[0,1])
        # print(f'Before disaster : A = {A.shape} B = {B.shape}')
        res = F.conv2d(A.unsqueeze(0).unsqueeze(0),B.unsqueeze(0).unsqueeze(0))
        res = res.squeeze() 
        if shape=='full':
            return res
        else:
            M,N = res.shape
            row = (M - cropX)//2
            col = (N - cropY)//2
            return res[row:row+cropX, col:col+cropY] 
    else:
        padCol = int((B.shape[1]*2 - 2)/2)
        padRow = int((B.shape[0]*2 - 2)/2)
        padding = (padCol, padCol, padRow, padRow)
        cropX, cropY = (A[:,:,0].shape[0]-B.shape[0]+1, A[:,:,0].shape[1]-B.shape[1]+1)
        B = torch.flip(B,[0,1])
        res = torch.zeros((int(A.shape[0] + B.shape[0]- 1), int(A.shape[1] + B.shape[1] - 1), A.shape[2]))
        for i in range(A.shape[2]):
            temp = F.pad(A[:,:,i], padding, value=0)
            res[:,:,i] = F.conv2d(temp.unsqueeze(0).unsqueeze(0),B.unsqueeze(0).unsqueeze(0))
            res[:,:,i] = res[:,:,i].squeeze(0).squeeze(0)
        if shape=='full':
            return res
        else:
            M,N,_ =res.shape
            row = (M - cropX)//2
            col = (N - cropY)//2
            return res[row:row+cropX, col:col+cropY,:]
            

def conv2Vector(u,v, A, shape):
    #pretty similar 
    #Lets forcefully make u a column vector
    if(u.shape[0] == 1):
        height = u.shape[1]
        u = u.reshape(height,1)
    if(v.shape[1] == 1):
        width = v.shape[0]
        v = v.reshape(1,width)
    cropX, cropY = (A.shape[0]-u.shape[0]+1, A.shape[1]-v.shape[1]+1)
    #Lets forcefully make v a row vector
    #number of rows to pad up and down for columnwise action
    padRow = int((u.shape[0]*2 - 2)/2)
    padCol = int((v.shape[1]*2 - 2)/2)
    padding = (padCol, padCol, padRow, padRow)
    A = F.pad(A,padding,value=0)
    u = torch.flip(u,[0])
    v = torch.flip(v,[1])
    
    initialConv = F.conv2d(A.unsqueeze(0).unsqueeze(0),u.unsqueeze(0).unsqueeze(0))
    res = F.conv2d(initialConv, v.unsqueeze(0).unsqueeze(0))
    res = res.squeeze()
    if(shape=='full'):
        return res
    else:
        M,N = res.shape
        row = (M - cropX)//2
        col = (N - cropY)//2
        # print(res[row:row+cropX,col:col+cropY])
        return res[row:row+cropX,col:col+cropY]

def fft(input):
    input = input.t()
    result = torch.fft.fft(input).t()
    return result

def fft2(input):
    if input.dim() == 2:
        # If input is already 2D, simply apply FFT
        input = input.t()
        input = torch.fft.fft(input).t()
        result = torch.fft.fft(input)
    elif input.dim() == 3:
        # If input is 3D, apply FFT along the last dimension for each 2D matrix
        result = torch.zeros((input.shape)).type(torch.complex64)
        # print(result.shape)
        for i in range(input.shape[2]):
            temp = input[:,:,i:i+1].squeeze()
            temp = temp.t()
            temp = torch.fft.fft(temp).t()
            result[:,:,i] = torch.fft.fft(temp)
            # print(result.dtype)
        return result
    else:
        raise ValueError("Input must be 2D or 3D")

    return result
def ifft(input):
    input = input.t()
    result = torch.fft.ifft(input).t()
    return result
def ifft2(input):
    if input.dim() == 2:
        input = input.t()
        input = torch.fft.ifft(input).t()
        result = torch.fft.ifft(input)
        return result
    elif input.dim() == 3:
        result = torch.zeros((input.shape)).type(torch.complex64)
        for i in range(input.shape[2]):
            temp = input[:,:,i].t()
            temp = torch.fft.ifft(temp).t()
            temp = torch.fft.ifft(temp)
            result[:,:,i] = temp
        return result
    else:
        raise ValueError("Input must be 2D or 3D")
#Expect sz to be list of dimensions containing two elements
#expect input to be 2 dimension
def psf2otf(input, sz):
    if input.ndim!=2:
        raise ValueError("Please use 2D data.")
    if not isinstance(sz,list):
        raise ValueError("Please use list for size parameter.")
    # if len(sz)!=2:
    #     raise ValueError("Please use list of length 2 for size parameter.")
    input = input.to(torch.float32)
    leftShift, topShift = int((input.shape[1]+2) / 2), int((input.shape[0]+2) / 2)
    leftShift -= 1
    topShift -= 1
    rightPad, bottomPad = int(sz[1] - input.shape[1]), int(sz[0] - input.shape[0])
    padding = (0,rightPad,0, bottomPad)
    input = F.pad(input, padding, value = 0)
    input = torch.roll(input,shifts=(-leftShift, -topShift), dims=[1,0])
    result = fft2(input)
    if len(sz) == 2:
        return result
    else:
        return result.unsqueeze(2)

def otf2psf(input, sz):
    if input.ndim!=2:
        raise ValueError("Please use 2D data")
    if not isinstance(sz,list):
        raise ValueError("Please use list for size parameter.")
    # if len(sz)!=2:
    #     raise ValueError("Please use list of length 2 for size parameter.")
    input = input.to(torch.float32)
    input = ifft2(input)
    shiftRight, shiftBottom = int((input.shape[1]+2)/2) , int((input.shape[0]+2)/2)
    centerX, centerY = shiftBottom, shiftRight
    shiftRight-=1
    shiftBottom-=1

    centerX -= 1
    centerY -= 1
    result = torch.roll(input, (shiftRight, shiftBottom), dims=[1,0])
    top = int(sz[0]/2)
    bottom = sz[0] - top - 1
    left =int(sz[1] / 2)
    right = sz[1] - left - 1
    if len(sz) == 2:
        return torch.real(result[centerX-top:centerX+bottom+1, centerY-left:centerY+right+1])
    else:
        return torch.real(result[centerX-top:centerX+bottom+1, centerY-left:centerY+right+1]).unsqueeze(2)

def custompad(tensor, pad):
    tensor2 = tensor.transpose(0,2)
    tensor2 = tensor2.to(torch.float32)
    tensor2 = F.pad(tensor2, (pad, pad, pad, pad), mode = "replicate")
    tensor2 = tensor2.transpose(0,2)
    return tensor2