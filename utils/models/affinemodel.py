import torch
from torch import nn
import torch.nn.functional as F


class AffineTransformModel(nn.Module):
    '''
    A model that can apply seperate rotations and translations in certain regions. 
    '''
    def __init__(self, rot=20., transX=0., transY=0., initializations=1, scale=True,):
        super().__init__()

        num_grids = 2*initializations

        self.transX_list = nn.Parameter(torch.tensor([transX] * num_grids))
        self.transY_list = nn.Parameter(torch.tensor([transY] * num_grids))
        if scale:
            print('not done')
            self.scaleX_list = nn.Parameter([torch.tensor([1., 0.])])
            self.scaleY_list = nn.Parameter([torch.tensor([0., 1.])])
        else: self.rot_list = nn.Parameter(torch.deg2rad(torch.tensor([rot]*num_grids)))


        self.theta = nn.Parameter(torch.zeros(num_grids, 2, 3), requires_grad = False)#.to(device)

        self.scale=scale
        self.relu = nn.ReLU()
        self.initializations = initializations


    def forward(self, x):
        '''
        x will be the mode derivatives, in shape (modes,2, nx,ny)

        #We have 2n grids for the affine model, where n is num initializations.
        #The first n are for the zerns. The second n are for the special mode.

        '''
        no_modes, s, sizex, sizey = x.size()
        x_zer = x[:-1].reshape(1,s*(no_modes-1),sizex,sizey).tile(self.initializations,1,1,1)
        x_special = x[-1].reshape(1,s,sizex,sizey).tile(self.initializations,1,1,1)

        theta = self.fill_theta()

        transformed_x_zer = self.transform(x_zer, theta[:self.initializations])
        transformed_x_special = self.transform(x_special, theta[self.initializations:])

        transformed_x = torch.cat([transformed_x_zer, transformed_x_special], dim=1).reshape(self.initializations, no_modes,s,sizex,sizey)

        return transformed_x

    def transform(self, x, theta):
        '''
        x is the modes, in shape (modes,2, nx,ny)
        theta is the transformation matrix, in shape (2,3)
        '''
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def fill_theta(self):

        theta = torch.zeros_like(self.theta)
                    
        if self.scale:
            theta[:,0,:2] = self.scaleX_list
            theta[:,1,:2] = self.scaleY_list
        else:
            rot_i = torch.cos(self.rot_list)  
            sin_i = torch.sin(self.rot_list)
            theta[:,0,0] = rot_i
            theta[:,0,1] = -sin_i
            theta[:,1,0] = sin_i
            theta[:,1,1] = rot_i

        theta[:,0,2] = self.transX_list
        theta[:,1,2] = self.transY_list

        return theta
    
    def set_theta(self, theta ):
        '''take rotation matrix and fill the rotation and translation lists'''
        
        self.transX_list.requires_grad = False
        self.transY_list.requires_grad = False
        self.transX_list = nn.Parameter(torch.tensor(theta[:,0,2].clone().detach()))
        self.transY_list = nn.Parameter(torch.tensor(theta[:,1,2].clone().detach()))
        self.transX_list.requires_grad = True
        self.transY_list.requires_grad = True
        
        if self.scale:
            # self.scaleX_list.requires_grad = False
            # self.scaleY_list.requires_grad = False

            self.scaleX_list = nn.Parameter(torch.tensor(theta[:,0,:2]))
            self.scaleY_list = nn.Parameter(torch.tensor(theta[:,1,:2]))
            self.scaleX_list.requires_grad = True
            self.scaleY_list.requires_grad = True
        else:
            self.rot_list.requires_grad = False
            angle = torch.atan2(theta[:,1,0], theta[:,0,0])
            self.rot_list = nn.Parameter(torch.tensor(angle))
            self.rot_list.requires_grad = True

        
