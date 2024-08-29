import os, sys, h5py
main_dir = os.path.abspath('../../')
# from utils.OverComplete.modes.zernike import Zernike
from utils.OverComplete.modes.zernike import Zernike
from utils.OverComplete.modes.vortex import Vortex
from utils.OverComplete.modes.pixel import Pixel
from utils.OverComplete.modes.affinemodel import AffineTransformModel
import numpy as np
import torch
from sklearn import linear_model


class ModalEvaluator:
    '''
    '''
    def __init__(self, size, n_zernike_rows, pixel_basis=False, zern_transform=True, microlens_pitch = 1, device='cuda'):   
    
        self.sizex, self.sizey = size
        xx_pad  = torch.linspace(-2,2,2*self.sizex)
        yy_pad = torch.linspace(-2,2,2*self.sizey)
        xx_pad, yy_pad = torch.meshgrid(xx_pad,yy_pad)

        dictionary, dictionary_grads = get_modes_and_derivs(offset=[0,0], xx=xx_pad, yy=yy_pad, n_zernike=n_zernike_rows, truncate_circle=False, pixel_basis = False)
        self.dictionary_grads = dictionary_grads.permute(1,0,2,3).to(device) / microlens_pitch# (modes,2,nx,ny) 
        self.dictionary = dictionary[:,None].to(device) # (modes,1,nx,ny)
        
        self.no_modes = len(dictionary)
        self.pixel_basis = pixel_basis
        if pixel_basis:
            #create pixel basis
            pix, pix_grads = get_pixel_basis(self.sizex, self.sizey)
            self.pix_grads = pix_grads.permute(1,0,2,3).to(device) / microlens_pitch# (modes,2,nx,ny)
            self.pix = pix[None].to(device) # (1,modes,nx,ny)

            self.pix = torch.nn.functional.pad(self.pix, (self.sizey//2,self.sizey - self.sizey//2,self.sizex//2,self.sizex -self.sizex//2))

            self.no_modes += len(pix)
        
        self.n_zernikes = n_zernike_rows * (n_zernike_rows + 1) // 2

        self.device = device

        self.zern_transform = zern_transform


    def fit(self, wavefront_derivs, epochs = 2000, lr=5e-3, l1_reg = 5e-3, fit_params = None):
        ''' 
        we can continue training with coefficients not None
        wavefront_derivs is (2, nx, ny)
        '''
        self.aff_model = AffineTransformModel(rot=0., transX=-0., transY=-0., scale=False).to(self.device)

        norm_factor = np.abs(np.nan_to_num(wavefront_derivs)).max()
        wavefront_derivs = wavefront_derivs / norm_factor

        t = wavefront_derivs.reshape(-1,1)
        t = torch.tensor(t).permute(1,0).float().to(self.device)
        notnans = ~torch.isnan(t)

        if fit_params is None:
            coefficients = torch.nn.Parameter(torch.rand(1,self.no_modes).to(self.device))
        else:
            coefficients = fit_params['coefficients']
            coefficients = torch.nn.Parameter(coefficients.to(self.device) / norm_factor)
            self.aff_model.set_theta(fit_params['theta'])

        history2 = {} 
        history2['loss'] = []

        all_params = list(self.aff_model.parameters()) + [coefficients]

        optimizer = torch.optim.Adam(all_params, lr=lr)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(1, epochs+1):

            optimizer.zero_grad()

            all_grads = self.aff_model(self.dictionary_grads)[...,self.sizex//2:3*self.sizex//2,self.sizey//2:3*self.sizey//2]

            if self.pixel_basis: all_grads = torch.cat((all_grads, self.pix_grads),dim=0)


            pred = (coefficients @ all_grads.reshape(self.no_modes,-1))

            mse = loss_fn(pred[notnans], t[notnans])

            reg = torch.norm(coefficients,1) / coefficients.size()[1]

            loss = mse + reg * l1_reg

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.aff_model.transX_list.data = torch.clamp(self.aff_model.transX_list.data, -.2, .2)
                self.aff_model.transY_list.data = torch.clamp(self.aff_model.transY_list.data, -.2, .2)

                if not self.zern_transform:
                    self.aff_model.rot_list.data[0] = 0
                    self.aff_model.transX_list.data[0] = 0
                    self.aff_model.transY_list.data[0] = 0
                
            print(f'Epoch {epoch}/{epochs}:, train mse: {mse:5.5g}, train reg: {reg*l1_reg:5.5g}',end='\r')

            history2['loss'].append(loss.item())

        theta = self.aff_model.fill_theta()
        coefficients = coefficients.detach() * norm_factor

        fit_params = {'coefficients': coefficients, 'theta': theta}

        return fit_params, history2
    

    def get_wavefront(self, fit_params, microlens_pitch = 150e-6):

        coefficients = fit_params['coefficients']
        theta = fit_params['theta']
        
        self.aff_model.set_theta(theta)

        all_grads = self.aff_model(self.dictionary_grads)[...,self.sizex//2:3*self.sizex//2,self.sizey//2:3*self.sizey//2]
        if self.pixel_basis: all_grads = torch.cat((all_grads, self.pix_grads),dim=0)
        pred = (coefficients @ all_grads.reshape(self.no_modes,-1))

        derivs = pred.reshape(2,self.sizex,self.sizey).detach().cpu().numpy()

        all_modes = self.aff_model(self.dictionary).permute(1,0,2,3)

        if self.pixel_basis: all_modes = torch.cat((all_modes, self.pix),dim=1)
        
        pred_wavefront = torch.sum(coefficients[:,:,None,None] * all_modes,dim=(0,1))[self.sizex//2:3*self.sizex//2,self.sizey//2:3*self.sizey//2].detach().cpu().numpy()

        pred_wavefront *= microlens_pitch/2

        return pred_wavefront, derivs




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


def get_modes_and_derivs(offset, xx, yy, n_zernike=3, truncate_circle=False, pixel_basis = True):
    x0,y0 = offset
# 
    zer, dzerdx, dzerdy = Zernike(n_zernike,xx,yy,x0,y0,truncate_circle=truncate_circle)

    vor, dvordx, dvordy = Vortex(2,xx,yy,x0,y0,truncate_circle=truncate_circle)

    pix, dpixdx, dpixdy = Pixel(xx,yy)

    if pixel_basis:
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

