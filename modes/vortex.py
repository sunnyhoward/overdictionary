import numpy as np

def Vortex(m, x, y, x0=0, y0=0, truncate_circle = True, normalize = True):
                
                
    nx, ny = np.size(x), np.size(y)
    xx,yy  = np.meshgrid(x,y)
    
    rr = ((xx-x0)**2+(yy-y0)**2)**.5

    psi = np.arctan2(yy - y0,xx-x0)
    # DpsiDy = (xx-x0)/(rr**2)
    # DpsiDx = -(yy-y0)/(rr**2)

    U = psi*m

    if normalize==True:
        U = U/np.max(U)
    
    dUdx, dUdy = np.gradient(U)

    if truncate_circle==True:
        U[rr>1]=0
        dUdx[rr>1]=0
        dUdy[rr>1]=0

    return(U[np.newaxis], dUdx[np.newaxis], dUdy[np.newaxis])