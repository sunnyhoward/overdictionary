import torch

def Pixel(xx, yy):
                

    nx, ny = xx.shape[0], xx.shape[1]


    U = torch.zeros((nx*ny,nx,ny))
    for i in torch.arange(nx):
        for k in torch.arange(ny):
            U[i*ny+k,i,k] = 1

    
    dUdx, dUdy = torch.gradient(U, axis=(1,2))

    # size = xx.shape
    # dUdy /= (size[0] - 1)/2
    # dUdx /= (size[1] - 1)/2

    return(U, dUdx, dUdy)