import torch
import numpy

def img_density(img_ary, coord):
    grid_density = -(img_ary - 255).T
    x, y = coord.numpy()

    # Determine grid cell boundaries 
    xf, yf = numpy.floor(coord).astype(int)
    xc, yc = numpy.ceil(coord).astype(int)
    
    # Bi-linear interpolation for density values at coordinate 
    density_xy_floor = (xc - x) * grid_density[xf, yf] + (x - xf) * grid_density[xc, yf]
    density_xy_ceil = (xc - x) * grid_density[xf, yc] + (x - xf) * grid_density[xc, yc]
    density_xy = (yc - y) * density_xy_floor + (y - yf) * density_xy_ceil

    return density_xy

if __name__ == '__main__':
    from scipy.misc import imread 
    from scipy.ndimage.filters import gaussian_filter
    import matplotlib.pyplot as plt
    
    img_ary = imread('probtorch-logo-bw.png', mode='L')
    img_ary = gaussian_filter(img_ary, sigma=1)
    # img_ary = numpy.array([[1.,2.],[3.,4.]]) # Dummy image for testing

    plt.matshow(img_ary)
    plt.show()

