import torch
import numpy

dtype = torch.FloatTensor
dtype_long = torch.LongTensor

def bilinearInterpolation(im, x, y):
    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1
    
    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1]-1)
    x1 = torch.clamp(x1, 0, im.shape[1]-1)
    y0 = torch.clamp(y0, 0, im.shape[0]-1)
    y1 = torch.clamp(y1, 0, im.shape[0]-1)
    
    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]
    
    wa = (x1.type(dtype) - x) * (y1.type(dtype) - y)
    wb = (x1.type(dtype) - x) * (y - y0.type(dtype))
    wc = (x - x0.type(dtype)) * (y1.type(dtype) - y)
    wd = (x - x0.type(dtype)) * (y - y0.type(dtype))

    return Ia*wa + Ib*wb + Ic*wc + Id*wd

if __name__ == '__main__':
    from scipy.misc import imread 
    from scipy.ndimage.filters import gaussian_filter
    import matplotlib.pyplot as plt
    
    img_ary = imread('probtorch-logo-bw.png', mode='L')
    img_ary = gaussian_filter(img_ary, sigma=0.1)
    grid_density = torch.FloatTensor(-(img_ary - 255).T)

    n = 100
    x = torch.rand(n) * grid_density.shape[0]
    y = torch.rand(n) * grid_density.shape[1]
    fxy = bilinearInterpolation(grid_density, x, y).numpy()
    # img_ary = numpy.array([[1.,2.],[3.,4.]]) # Dummy image for testing

    plt.matshow(img_ary)
    plt.scatter(x, y, c='r')
    plt.scatter(x[(fxy > 0.5).nonzero()], y[(fxy > 0.5).nonzero()], c='b')
    plt.show()

