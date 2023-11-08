import torch
import torch.nn.functional as F

def dark_channel(I, patch_size):
    M, N, C = I.shape
    J = torch.zeros(M, N)  # Create an empty matrix for J
    J_index = torch.zeros(M, N)  # Create an empty index matrix

    # Test if patch size has an odd number
    if patch_size % 2 == 0:
        raise ValueError("Invalid Patch Size: Only odd-numbered patch sizes are supported.")

    # Pad the original image
    I = F.pad(I, (patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2), mode='replicate')

    # Compute the dark channel
    for m in range(M):
        for n in range(N):
            patch = I[m:(m+patch_size), n:(n+patch_size), :]
            tmp = torch.min(patch, dim=2)[0]
            tmp_val, tmp_idx = torch.min(tmp.view(-1))
            J[m, n] = tmp_val
            J_index[m, n] = tmp_idx

    return J, J_index
