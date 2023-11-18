import torch

opt_fft_size_LUT = None
#Expected input is a list of numbers representing the dimension of the image
def opt_fft_size(n):
    LUT_size = 4097
    
    global opt_fft_size_LUT
    if(opt_fft_size_LUT is None):
        print('generating optimal fft size Look up table')
        opt_fft_size_LUT = [0] * LUT_size
        e2 = 1
        while e2 < LUT_size:
            e3 = e2
            while e3 < LUT_size:
                e5 = e3
                while e5 < LUT_size:
                    e7 = e5
                    while e7 < LUT_size:
                        if e7 < LUT_size:
                            opt_fft_size_LUT[e7 ] = e7
                        if e7 * 11 < LUT_size:
                            opt_fft_size_LUT[e7 * 11 ] = e7 * 11
                        if e7 * 13 < LUT_size:
                            opt_fft_size_LUT[e7 * 13 ] = e7 * 13
                        e7 *= 7
                    e5 *= 5
                e3 *= 3
            e2 *= 2

        nn = 0
        for i in range(LUT_size - 1, 0, -1):
            if opt_fft_size_LUT[i] != 0:
                nn = i
            else:
                opt_fft_size_LUT[i] = nn
    # let m be a list as well
    m = [0] * len(n)
    for c in range(len(n)):
        nn = n[c]
        if nn < LUT_size:
            m[c] = opt_fft_size_LUT[nn]
        else:
            m[c] = -1
    # print(f'yolo {m}')
    return m