import torch

def conjgrad(x, b, maxIt, tol, Ax_func, func_param, visfunc=None):
    # print(f'inside conjgrad b shape {b.shape}')
    # x = Ax_func(x, func_param)
    # print(f'inside conjgrad ax_func result shape {x.shape}')
    r = b - Ax_func(x, func_param)
    p = r
    rsold = torch.sum(r * r)

    for iter in range(1, maxIt + 1):
        Ap = Ax_func(p, func_param)
        alpha = rsold / torch.sum(p * Ap)
        x = x + alpha * p

        if visfunc is not None:
            visfunc(x, iter, func_param)

        r = r - alpha * Ap
        rsnew = torch.sum(r * r)

        if torch.sqrt(rsnew) < tol:
            break

        p = r + rsnew / rsold * p
        rsold = rsnew

    return x