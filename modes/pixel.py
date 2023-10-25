import numpy as np

def Pixel(x, y):
                

    nx, ny = np.size(x), np.size(y)
    xx,yy  = np.meshgrid(x,y)

    rr = ((xx)**2+(yy)**2)**.5


    U = np.zeros((nx*ny,nx,ny))
    for i in np.arange(nx):
        for k in np.arange(ny):
            U[i*ny+k,i,k] = 1

    
    dUdx, dUdy = np.gradient(U, axis=(1,2))


    return(U, dUdx, dUdy)