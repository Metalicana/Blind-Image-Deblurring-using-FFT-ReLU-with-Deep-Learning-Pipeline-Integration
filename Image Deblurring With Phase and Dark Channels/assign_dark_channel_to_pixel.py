import torch
import torch.nn.functional as F
from misc import custompad
#from dark_channel import dark_channel

def assign_dark_channel_to_pixel(S, dark_channel_refine, dark_channel_index, patch_size):
    M, N, C = S.size()
    padsize = patch_size // 2
    S_padd = custompad(S,padsize)
    
    for m in range(M):
        for n in range(N):
            patch = S_padd[m:m+patch_size, n:n+patch_size, :]

            if not torch.equal(torch.min(patch), dark_channel_refine[m, n]):
                patch[dark_channel_index[m, n]] = dark_channel_refine[m, n]

            for cc in range(C):
                S_padd[m:m+patch_size, n:n+patch_size, cc] = patch[:, :, cc]

    outImg = S_padd[padsize:-padsize, padsize:-padsize, :]

    # Boundary processing
    outImg[0:padsize, :, :] = S[0:padsize, :, :]
    outImg[-padsize:, :, :] = S[-padsize:, :, :]
    outImg[:, 0:padsize, :] = S[:, 0:padsize, :]
    outImg[:, -padsize:, :] = S[:, -padsize:, :]

    return outImg

'''
tensor3d = torch.tensor([
    [
        [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
    ],
    [
        [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24],
    ]
],dtype=torch.float32)
print(tensor3d)
v, idx = dark_channel(tensor3d, 3)
u = v
print(assign_dark_channel_to_pixel(tensor3d,u,idx,3))
'''