import os, sys, h5py
main_dir = os.path.abspath('../')
sys.path.append(main_dir)
from utils.modes.zernike import Zernike
from utils.modes.vortex import Vortex
from utils.modes.pixel import Pixel
import numpy as np
import torch
from sklearn import linear_model


def get_modes_and_derivs(offset, xx, yy, n_zernike=3, truncate_circle=False, pixel_basis = True, special_mode = Vortex):
    # simply return the mode and derivs for zernikes, vortex and pixel basis, for a given offset.
    x0,y0 = offset
# 
    zer, dzerdx, dzerdy = Zernike(n_zernike,xx,yy,x0,y0,truncate_circle=truncate_circle)

    special, dspecialdx, dspecialdy = special_mode(xx,yy,x0,y0,truncate_circle=truncate_circle)

    if pixel_basis:
        pix, dpixdx, dpixdy = Pixel(xx,yy)
        modes = torch.concat((zer,special,pix),axis=0)
        dmodesdx = torch.concat((dzerdx,dspecialdx,dpixdx),axis=0)
        dmodesdy = torch.concat((dzerdy,dspecialdy,dpixdy),axis=0)
    else:
        modes = torch.concat((zer,special),axis=0)
        dmodesdx = torch.concat((dzerdx,dspecialdx),axis=0)
        dmodesdy = torch.concat((dzerdy,dspecialdy),axis=0)

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



def zonal_reconstruction(phase_grad_x, phase_grad_y, phase_grad_x_std, phase_grad_y_std, microlens_pitch=150e-6):
    '''
    both derivatives have shape (n_wl, nx, ny). Perform zonal reconstruction to get the phase.

    '''
    n_wl, nx, ny = phase_grad_x.shape

    derivs = np.concatenate(( phase_grad_x.reshape(n_wl,-1), phase_grad_y.reshape(n_wl,-1)),axis=-1)
    derivs_var = np.concatenate(( phase_grad_x_std.reshape(n_wl,-1), phase_grad_y_std.reshape(n_wl,-1)),axis=-1)**2
    
    #make deriv matrices
    Dx = np.zeros((nx * ny, nx * ny))
    Dy = np.zeros((nx * ny, nx * ny))

    Ax = np.zeros((nx * ny, nx * ny))
    Ay = np.zeros((nx * ny, nx * ny))

    # Fill D with sub-matrices for x and y derivatives
    for i in range(nx):
        for j in range(ny):
            index = i * ny + j
            # Derivative in x (except for the last column)
            if j < ny - 1:
                Dx[index, index + 1] = 1 / microlens_pitch
                Dx[index, index] = -1 / microlens_pitch
                Ax[index, index + 1] = .5
                Ax[index, index] = .5
            # Derivative in y (except for the last row)
            if i < nx - 1:
                Dy[index, index] = -1 / microlens_pitch
                Dy[index, index + ny] = 1 / microlens_pitch
                Ay[index, index] = .5
                Ay[index, index + ny] = .5

    D = np.vstack([Dy, Dx])
    A = np.vstack([np.hstack([Ay, np.zeros_like(Ay)]), np.hstack([np.zeros_like(Ay), Ax])])

    Dinv = np.linalg.pinv(D)

    phase = (Dinv @ A @ np.nan_to_num(derivs).T).T.reshape(*phase_grad_x.shape)
    phase_std = np.sqrt(((Dinv @ A)**2 @ np.nan_to_num(derivs_var).T).T.reshape(*phase_grad_x.shape))

    phase[np.isnan(phase_grad_x)] = np.nan
    phase_std[np.isnan(phase_grad_x)] = np.nan

    return phase, phase_std