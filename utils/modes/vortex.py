import torch

def Vortex(xx, yy, x0=0, y0=0, truncate_circle = True):
                
                    
    rr = ((xx-x0)**2+(yy-y0)**2)**.5

    psi = torch.arctan2(yy - y0,xx-x0)


    U = psi
    dUdy = (xx-x0)/(rr**2)
    dUdx = -(yy-y0)/(rr**2)

    
    # dUdx, dUdy = torch.gradient(U)

    if truncate_circle==True:
        rr_nan = ((xx)**2+(yy)**2)**.5
        U[rr_nan>1]=torch.nan
        dUdx[rr_nan>1]=torch.nan
        dUdy[rr_nan>1]=torch.nan

    # scale the derivatives so microlens pitch = 1. Currently microlens pitch is 
    old_pitch = (xx[1,1] - xx[0,0]).abs()
    
    dUdy *= old_pitch
    dUdx *= old_pitch

    return(U.unsqueeze(0), dUdx.unsqueeze(0), dUdy.unsqueeze(0))