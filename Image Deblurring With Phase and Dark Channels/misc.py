import torch
import torch.fft as fft
import torch.nn.functional as F
from matplotlib import pyplot as plt
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
    padCol = int((B.shape[1]*2 - 2)/2)
    padRow = int((B.shape[0]*2 - 2)/2)
    cropX, cropY = (A.shape[0]-B.shape[0]+1, A.shape[1]-B.shape[1]+1)
    
    padding = (padCol, padCol, padRow, padRow)
    A = F.pad(A,padding, value=0)
    
    B = torch.flip(B,[0,1])
    res = F.conv2d(A.unsqueeze(0).unsqueeze(0),B.unsqueeze(0).unsqueeze(0))
    res = res.squeeze() 
    if shape=='full':
        return res
    else:
        M,N = res.shape
        row = (M - cropX)//2
        col = (N - cropY)//2
        return res[row:row+cropX, col:col+cropY] 

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
    input = input.t()
    input = torch.fft.fft(input).t()
    result = torch.fft.fft(input)
    return result
def ifft(input):
    input = input.t()
    result = torch.fft.ifft(input).t()
    return result
def ifft2(input):
    input = input.t()
    input = torch.fft.ifft(input).t()
    result = torch.fft.ifft(input)
    return result

#Expect sz to be list of dimensions
#expect input to be whatever dimension
def psf2otf(input, sz):
    leftShift, topShift = int((input.shape[1]+2) / 2), int((input.shape[0]+2) / 2)
    leftShift -= 1
    topShift -= 1
    rightPad, bottomPad = int(sz[1] - input.shape[1]), int(sz[0] - input.shape[0])
    padding = (0,rightPad,0, bottomPad)
    input = F.pad(input, padding, value = 0)
    input = torch.roll(input,shifts=(-leftShift, -topShift), dims=[1,0])
    result = fft2(input)
    return result

def otf2psf(input, sz):
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
    return torch.real(result[centerX-top:centerX+bottom+1, centerY-left:centerY+right+1])