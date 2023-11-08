import torch
import torch.nn.functional as F

def assign_dark_channel_to_pixel(S, dark_channel_refine, dark_channel_index, patch_size):
    M, N, C = S.size()
    padsize = patch_size // 2
    S_padd = F.pad(S, (padsize, padsize, padsize, padsize), mode='edge')

    for m in range(M):
        for n in range(N):
            patch = S_padd[m:m+patch_size, n:n+patch_size, :]

            if not torch.equal(torch.min(patch, dim=(0, 1))[0], dark_channel_refine[m, n]):
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
