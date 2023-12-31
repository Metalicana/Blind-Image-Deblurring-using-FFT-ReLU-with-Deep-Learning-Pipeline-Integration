import torch

from misc import psf2otf, fft, fft2, ifft2
import math
def sign(n):
    if n==0: return n
    else: return n/abs(n)

def computeDenominator(y, k):
    #print(y.shape)
    otfk = psf2otf(k, list(y.size()))
    
    Nomin1 = torch.conj(otfk) * fft2(y)
    Denom1 = torch.abs(otfk)**2
    filtx = torch.tensor([[1,-1]])
    filty = filtx.t()
    # print(f'filtx is {filtx.shape}')
    Denom2 = torch.abs(psf2otf(filtx,list(y.size())))**2 + \
            torch.abs(psf2otf(filty,list(y.size())))**2
    return Nomin1, Denom1, Denom2

def row_col_diff_pair(data):
    dims = []
    dims = data.shape
    if(len(dims)!=2):
        raise ValueError("Please use 2D data.")
    # Calculate row-wise differences (differences along columns)
    # row_diff = data[1:, :] - data[:-1, :]
    row_diff = torch.diff(data, 1, 1)
    # print(f'row_diff is {row_diff.shape}')
    a = data[:,0]-data[:,-1]
    a = a.view(a.shape[0],1)
    # print(f'a is {a.shape}')
    row_diff = torch.cat([row_diff,a],dim = 1)
    # print(f'row_diff is {row_diff.shape}')
    # Calculate column-wise differences (differences along rows)
    column_diff = torch.diff(data, 1, 0)
    # print(f'column_diff is {column_diff.shape}')
    a = data[0,:]-data[-1,:]
    a = a.view(1,a.shape[0])
    # print(f'a is {a.shape}')
    
    # first_last_column_diff = data[:, 0] - data[:, -1]
    # first_last_column_diff = first_last_column_diff.view(-1,1)

    column_diff = torch.cat((column_diff,a),dim = 0)
    # Print the row-wise and column-wise differences
    return row_diff, column_diff


def deblurring_adm_aniso(B, k, lambda_val, alpha):
    beta = 1.0 / lambda_val
    beta_rate = 2.0 * math.sqrt(2.0)
    beta_min = 0.001

    m, n = B.shape
    I = B.clone()

    if k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
        raise ValueError("Blur kernel k must be odd-sized.")

    Nomin1, Denom1, Denom2 = computeDenominator(B, k)
    #print(Denom2)

    Ix, Iy = row_col_diff_pair(I)
    
    while beta > beta_min:
        gamma = 1.0 / (2.0 * beta)
        Denom = Denom1 + gamma*Denom2

        if alpha == 1:
            Wx = torch.abs(Ix) - beta*lambda_val
            Wx[Wx<0] = 0
            Wx = Wx * torch.sign(Ix)
            Wy = torch.abs(Iy) - beta*lambda_val
            Wy[Wy<0] = 0
            Wy = Wy * torch.sign(Iy)
            #print("WWYY")
            #print(Wy)
            #print(Wx)
            #Wx = torch.max(torch.abs(Ix) - beta * lambda_val, torch.tensor(0, dtype=Ix.dtype, device=Ix.device)) * torch.sign(Ix)
            #Wy = torch.max(torch.abs(Iy) - beta * lambda_val, torch.tensor(0, dtype=Iy.dtype, device=Iy.device)) * torch.sign(Iy)
            
        else:
            raise ValueError("Implementation unavailable for values of alpha not equal to 1.")
        # Wxx = [Wx(:,n) - Wx(:, 1), -diff(Wx,1,2)]; 
        # Wxx = Wxx + [Wy(m,:) - Wy(1, :); -diff(Wy,1,1)]; 
        a = Wx[:,-1]-Wx[:,0]
        # print(f'a is {a.shape}')
        b = -torch.diff(Wx,1,1)
        # print(f'b is {b.shape}')
        Wxx = torch.cat((a.view(a.shape[0],1),b), dim = 1)
        # print(f'Wxx is {Wxx.shape}')
        #print("WXX")
        #print(Wxx)
        a = Wy[-1,:]-Wy[0,:]
        b = -torch.diff(Wy,1,0)
        Wxx = Wxx + torch.cat((a.view(1,a.shape[0]),b))
        #print("WYY")
        #print(Wxx)
        #Wxx = torch.cat([(Wx[:, -1] - Wx[:, 0]).unsqueeze(1), -torch.cat([Wx[:, 1:] - Wx[:, :-1], Wx[:, 0].unsqueeze(1) - Wx[:, -1].unsqueeze(1)], dim=1)], dim=1)
        #Wxx += torch.cat([(Wy[-1, :] - Wy[0, :]).unsqueeze(0), -torch.cat([Wy[1:] - Wy[:-1], (Wy[0, :] - Wy[-1, :]).unsqueeze(0)], dim=0)], dim=0)

        Fyout = (Nomin1 + gamma * fft2(Wxx)) / Denom
       
        #print(Fyout)
        I = torch.real(ifft2(Fyout))
        #print("I")
        #print(I)
        Ix, Iy = row_col_diff_pair(I)
        beta = beta / beta_rate

    return I

# Example usage:
# B, k, lambda_val, alpha = ...  # Set your inputs here
# deblurred_image = deblurring_adm_aniso(B, k, lambda_val, alpha)
img = torch.arange(1, 10*10 + 1).reshape(10, 10)
ker = torch.arange(1, 5*5 + 1).reshape(5,5)

t = deblurring_adm_aniso(img,ker,0.002,1)

#print(t)