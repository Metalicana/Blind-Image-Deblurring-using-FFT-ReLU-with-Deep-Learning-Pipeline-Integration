import torch
from misc import dst,idst
#Expecting an img of shape H, W, num_channels
#Expecting an img_size a list 
def wrap_boundary_liu(img, img_size):
    H, W, Ch = img.shape
    H_w = img_size[0] - H
    W_w = img_size[1] - W

    ret = torch.zeros((img_size[0], img_size[1], Ch), dtype=img.dtype)

    for ch in range(Ch):
        alpha = 1
        HG = img[:, :, ch]

        r_A = torch.zeros((alpha*2+H_w, W), dtype=img.dtype)
        row_r_A, col_r_A = r_A.shape
        
        r_A[0:alpha, :] = HG[HG.shape[0]-alpha:HG.shape[0], :]
        r_A[row_r_A-alpha:row_r_A, :] = HG[0:alpha, :]
        a = (torch.arange(1, H_w+1) - 1) / (H_w - 1)
        r_A[alpha:row_r_A-alpha, 0] = (1 - a) * r_A[alpha-1, 0] + a * r_A[row_r_A-alpha, 0]
        r_A[alpha:row_r_A-alpha, col_r_A-1 ] = (1 - a) * r_A[alpha-1, col_r_A-1] + a * r_A[row_r_A-alpha, col_r_A-1]

        A2 = solve_min_laplacian(r_A[alpha-1:row_r_A-alpha+1, :])
        r_A[alpha-1:row_r_A-alpha+1, :] = A2
        A = r_A

        r_B = torch.zeros(H, alpha*2+W_w, dtype=img.dtype)
        row_r_B, col_r_B = r_B.shape
        r_B[:, 0:alpha] = HG[:, HG.shape[1]-alpha:HG.shape[1]]
        r_B[:, col_r_B-alpha:col_r_B] = HG[:, 0:alpha]
        a = (torch.arange(1, W_w+1) - 1) / (W_w - 1)
        #this line might be wrong
        r_B[0, alpha:col_r_B-alpha] = (1 - a) * r_B[0, alpha-1] + a * r_B[0, col_r_B-alpha]
        r_B[row_r_B-1, alpha:col_r_B-alpha] = (1 - a) * r_B[row_r_B-1, alpha-1] + a * r_B[row_r_B-1, col_r_B-alpha]

        B2 = solve_min_laplacian(r_B[:, alpha-1:col_r_B-alpha+1])
        r_B[:, alpha-1:col_r_B-alpha+1] = B2
        B = r_B

        r_C = torch.zeros(alpha*2+H_w, alpha*2+W_w, dtype=img.dtype)
        row_r_C, col_r_C = r_C.shape
        r_C[0:alpha, :] = B[row_r_B-alpha:row_r_B, :]
        r_C[row_r_C-alpha:row_r_C, :] = B[0:alpha, :]
        r_C[:, 0:alpha] = A[:, col_r_A-alpha:col_r_A]
        r_C[:, col_r_C-alpha:col_r_C] = A[:, 0:alpha]

        C2 = solve_min_laplacian(r_C[0:row_r_C-alpha+1, 0:col_r_C-alpha+1])
        r_C[0:row_r_C-alpha+1, alpha-1:col_r_C-alpha+1] = C2
        C = r_C

        A = A[0:row_r_A-alpha-1, :]
        B = B[:, alpha:col_r_B-alpha]
        C = C[alpha:row_r_C-alpha, alpha:col_r_C-alpha]
        # print(A.shape, B.shape, C.shape, img[:,:,ch].shape)
        t1 = torch.cat((img[:, :, ch], B), dim=1)
        # print(t1.shape)
        t2 = torch.cat((A, C), dim=1)
        ret[:,:,ch] = torch.cat((t1,t2),dim=0)

    return ret

def solve_min_laplacian(boundary_image):
    H, W = boundary_image.shape
    f = torch.zeros((H, W), dtype=boundary_image.dtype)

    boundary_image[1:-1, 1:-1] = 0
    j = torch.arange(1,H-1)
    k = torch.arange(1,W-1)
    f_bp = torch.zeros((H,W))
    #select the submatrix and assign
    f_bp[1:-1,1:-1] = -4*boundary_image[1:-1,1:-1] + boundary_image[1:-1,2:] + boundary_image[1:-1,:-2] + boundary_image[2:,1:-1] + boundary_image[:-2,1:-1]
    # f_bp = -4 * boundary_image + boundary_image[:, 1:] + boundary_image[:, :-1] + boundary_image[1:, :] + boundary_image[:-1, :]

    f1 = f - f_bp

    f2 = f1[1:-1, 1:-1]
    # print(f'f1shape {f1.shape}')
    tt = dst(f2)
    # print(f'tt shape {tt.shape}')
    f2sin = dst(tt.transpose(0, 1)).transpose(0, 1)

    x, y = torch.meshgrid(torch.arange(1, W-1), torch.arange(1, H-1))
    denom = (2 * torch.cos(torch.pi * x / (W-1)) - 2) + (2 * torch.cos(torch.pi * y / (H-1)) - 2)
    denom = denom.t()
    # print(f'denom shape {denom.shape}')
    f3 = f2sin / denom

    tt = idst(f3)
    img_tt = idst(tt.transpose(0, 1)).transpose(0, 1)

    img_direct = boundary_image.clone()
    img_direct[1:H-1, 1:W-1] = 0
    img_direct[1:H-1, 1:W-1] = img_tt

    return img_direct

# DST and IDST functions (you can use PyTorch's dst and idst functions if available)



