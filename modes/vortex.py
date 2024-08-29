import numpy as np
import torch

def Vortex(m, xx, yy, x0=0, y0=0, truncate_circle = True):
                
                    
    rr = ((xx-x0)**2+(yy-y0)**2)**.5

    psi = torch.arctan2(yy - y0,xx-x0)


    U = psi*m
    dUdy = m*(xx-x0)/(rr**2)
    dUdx = -m*(yy-y0)/(rr**2)

    
    # dUdx, dUdy = torch.gradient(U)

    if truncate_circle==True:
        U[rr>1]=0
        dUdx[rr>1]=0
        dUdy[rr>1]=0

    size = xx.shape
    dUdy /= (size[0] - 1)/2
    dUdx /= (size[1] - 1)/2

    return(U.unsqueeze(0), dUdx.unsqueeze(0), dUdy.unsqueeze(0))