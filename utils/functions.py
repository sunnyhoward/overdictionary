import os, sys, h5py
main_dir = os.path.abspath('../')
sys.path.append(main_dir)
# from utils.OverComplete.modes.zernike import Zernike
from utils.modes.zernike import Zernike
from utils.modes.vortex import Vortex
from utils.modes.pixel import Pixel
import numpy as np
import torch
from sklearn import linear_model




def get_modes_and_derivs(offset, xx, yy, n_zernike=3, truncate_circle=False, pixel_basis = True):
    # simply return the mode and derivs for zernikes, vortex and pixel basis, for a given offset.
    x0,y0 = offset
# 
    zer, dzerdx, dzerdy = Zernike(n_zernike,xx,yy,x0,y0,truncate_circle=truncate_circle)

    vor, dvordx, dvordy = Vortex(1,xx,yy,x0,y0,truncate_circle=truncate_circle)

    if pixel_basis:
        pix, dpixdx, dpixdy = Pixel(xx,yy)
        modes = torch.concat((zer,vor,pix),axis=0)
        dmodesdx = torch.concat((dzerdx,dvordx,dpixdx),axis=0)
        dmodesdy = torch.concat((dzerdy,dvordy,dpixdy),axis=0)
    else:
        modes = torch.concat((zer,vor),axis=0)
        dmodesdx = torch.concat((dzerdx,dvordx),axis=0)
        dmodesdy = torch.concat((dzerdy,dvordy),axis=0)

    mode_gradients = torch.stack((dmodesdx,dmodesdy),axis=0)

    return modes, mode_gradients


def get_pixel_basis(sizex,sizey):
    xx  = torch.linspace(-1,1,sizex)
    yy = torch.linspace(-1,1,sizey)
    xx, yy = torch.meshgrid(xx,yy)
    pix, dpixdx, dpixdy = Pixel(xx,yy)
    pix_gradients = torch.stack((dpixdx,dpixdy),axis=0)
    return pix, pix_gradients



def LassoFit(sampled_gradient, mode_gradients, alpha = .01):
    '''
    sampled gradient is (2,nx,ny)
    mode_gradients is (n_modes,2,nx,ny)
    this cannot incorperate decentered vortices.
    '''
    t = sampled_gradient.reshape(-1,1)
    u = mode_gradients.permute(0,2,3,1).reshape(-1,mode_gradients.shape[1])

    u=u[(~torch.isnan(t)).bool()[:,0]]
    t=t[~torch.isnan(t)]


    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(u,t)
    result_vector = clf.coef_

    wavefront_grad_prediction = (mode_gradients.permute(0,2,3,1) @ result_vector)

    return wavefront_grad_prediction, result_vector



def clean_gradient(sampled_gradient, threshold = .6):
    '''sampled_gradient is (2,wl,x,y)''' 

    nans = np.isnan(sampled_gradient)
    ffted = np.fft.fftshift(np.fft.fft2(np.nan_to_num(sampled_gradient), axes=(-2,-1)),axes=(-2,-1))
    x = np.linspace(-1,1,ffted.shape[-1])
    y = np.linspace(-1,1,ffted.shape[-2])
    xx, yy = np.meshgrid(x, y)
    mask = (xx)**2 + (yy)**2 < threshold**2
    ffted *= mask[None,None]

    sampled_gradient_clean = np.real(np.fft.ifft2(np.fft.ifftshift(ffted,axes=(-2,-1)), axes=(-2,-1)))

    sampled_gradient_clean[nans] = np.nan
    return sampled_gradient_clean