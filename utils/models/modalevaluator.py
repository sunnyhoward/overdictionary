import os, sys, h5py
main_dir = os.path.abspath('../../')
sys.path.append(main_dir)
from utils.models.affinemodel import AffineTransformModel
from utils.functions import *
import numpy as np
import torch
from sklearn import linear_model


class ModalEvaluator:
    '''
    This is the master model that fits the modal coefficients and affine transform coefficients to the measured derivatives.

    Here you say how many zernike rows to fit, whether the pixel basis should be included (for high frequency). Also use the 
    'zern_transform' parameter to decide if the affine model should effect the zernikes. 
    
    Here a vortex is always included in the dictionary
    '''

    def __init__(self, size, n_zernike_rows, pixel_basis=False, zern_transform=True, microlens_pitch = 1, device='cuda'):   
    
        self.sizex, self.sizey = size
        xx_pad  = torch.linspace(-2,2,2*self.sizex)
        yy_pad = torch.linspace(-2,2,2*self.sizey)
        xx_pad, yy_pad = torch.meshgrid(xx_pad,yy_pad)

        # create the dictionary
        dictionary, dictionary_grads = get_modes_and_derivs(offset=[0,0], xx=xx_pad, yy=yy_pad, n_zernike=n_zernike_rows,               truncate_circle=False, pixel_basis = False)
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
        fit our modal coefficients and affine parameters to the wavefront derivatives.
        wavefront_derivs is (2, nx, ny)

        we can continue training with fit_params not None
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
        '''
        return the predicted wavefront and derivatives
        '''

        coefficients = fit_params['coefficients']
        theta = fit_params['theta']
        
        self.aff_model.set_theta(theta)

        all_grads = self.aff_model(self.dictionary_grads)[...,self.sizex//2:3*self.sizex//2,self.sizey//2:3*self.sizey//2]
        if self.pixel_basis: all_grads = torch.cat((all_grads, self.pix_grads),dim=0)
        pred = (coefficients @ all_grads.reshape(self.no_modes,-1))

        pred_derivs = pred.reshape(2,self.sizex,self.sizey).detach().cpu().numpy()

        all_modes = self.aff_model(self.dictionary).permute(1,0,2,3)

        if self.pixel_basis: all_modes = torch.cat((all_modes, self.pix),dim=1)
        
        pred_wavefront = torch.sum(coefficients[:,:,None,None] * all_modes,dim=(0,1))[self.sizex//2:3*self.sizex//2,self.sizey//2:3*self.sizey//2].detach().cpu().numpy()

        pred_wavefront *= microlens_pitch/2

        return pred_wavefront, pred_derivs