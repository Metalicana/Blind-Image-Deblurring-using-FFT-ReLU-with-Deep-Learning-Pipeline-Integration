import torch
import torch.nn.functional as F
from misc import custompad
def dark_channel(I, patch_size):
    M, N, C = I.shape
    J = torch.zeros(M, N)  # Create an empty matrix for J
    J_index = torch.zeros(M, N)  # Create an empty index matrix
    # Test if patch size has an odd number
    if patch_size % 2 == 0:
        raise ValueError("Invalid Patch Size: Only odd-numbered patch sizes are supported.")

    # Pad the original image
    #I = F.pad(I, (patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2), mode='replicate')
    I  = custompad(I,patch_size//2)
    # Compute the dark channel
    for m in range(M):
        for n in range(N):
            patch = I[m:(m+patch_size), n:(n+patch_size), :]
            tmp = torch.min(patch)
            indices = (patch == tmp).nonzero()
            J[m, n] = tmp
            J_index[m,n] = indices[0][2]

    return J, J_index
'''
depth = 2
rows = 3
columns = 4


tensor3d = torch.tensor([
    [
        [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
    ],
    [
        [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24],
    ]
],dtype=torch.float32)

v, idx = dark_channel(tensor3d, 3)
print(v)
print(idx)
'''