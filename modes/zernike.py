import numpy as np
import torch

def Zernike(n_zernike_row, xx, yy, x0=0, y0=0, truncate_circle = False):
    """
    Returns the Zernike polynomials up to order n_max, and their derivatives.

    Parameters
    ----------
    n_max : int
        The maximum order of the Zernike polynomials.
    x : array_like
        The x-coordinates of the grid.
    y : array_like
        The y-coordinates of the grid.
    truncate_circle : bool, optional
        Whether to truncate the polynomials at the unit circle.

    Returns
    -------
    U : array_like
        The Zernike polynomials.
    dUdx : array_like
        The derivative of the Zernike polynomials with respect to x.
    dUdy : array_like
        The derivative of the Zernike polynomials with respect to y.

    Examples
    --------
    >>> x = np.linspace(-1,1,100)
    >>> y = np.linspace(-1,1,100)
    >>> U, dUdx, dUdy = Zernike(3, x, y)
    """

    n_max = n_zernike_row #find_row(n_zernike)+1


    nx, ny = xx.shape[0], xx.shape[1]
    
    xx = xx - x0
    yy = yy - y0

    rr = (xx**2+yy**2)**.5
    if truncate_circle==True:
        xx[rr>1]=0
        yy[rr>1]=0

    (m_max, n_max) = n_max, n_max

    U = torch.zeros([m_max, n_max, ny, nx])
    dUdx = torch.zeros_like(U)
    dUdy = torch.zeros_like(U)

    # (0,0)
    U[0,0,:,:] = torch.ones_like(xx)
    # U[0,0,rr>1] = 0
    dUdx[0,0,:,:] = torch.zeros_like(xx)
    dUdy[0,0,:,:] = torch.zeros_like(xx)

    # (1,0)
    U[1,0,:,:] = yy
    dUdx[1,0,:,:] = torch.zeros_like(xx)
    dUdy[1,0,:,:] = torch.ones_like(xx)
    # dUdy[1,0,rr>1] = 0

    # (1,1)
    U[1,1,:,:] = xx
    dUdx[1,1,:,:] = torch.ones_like(xx)
    # dUdx[1,1,rr>1] = 0
    dUdy[1,1,:,:] = torch.zeros_like(xx)

    for n in range(2,n_max):
        for m in range(0,n+1):
            #print(n,m)
            U[n,m] = xx*U[n-1,m] + yy*U[n-1,n-1-m] + xx*U[n-1, m-1] - yy*U[n-1, n-m] - U[n-2,m-1]
            dUdx[n,m] = n*U[n-1,m] + n*U[n-1,m-1] + dUdx[n-2,m-1]
            dUdy[n,m] = n*U[n-1,n-m-1] - n*U[n-1,n-m] + dUdy[n-2,m-1]

            if m==0:
                U[n,m,:,:] = xx*U[n-1,0,:,:]+yy*U[n-1,n-1,:,:]
                dUdx[n,m,:,:] = n*U[n-1,0,:,:]
                dUdy[n,m,:,:] = n*U[n-1,n-1,:,:]
            if m==n:
                U[n,m] = xx*U[n-1,n-1]-yy*U[n-1,0]
                dUdx[n,m] = n*U[n-1,n-1]
                dUdy[n,m] = -n*U[n-1,0]

            if n%2==1 and m==(n-1)//2:
                U[n,m] = yy*U[n-1,n-1-m] + xx*U[n-1,m-1] - yy*U[n-1, n-m] - U[n-2,m-1]
                dUdx[n,m] = n*U[n-1,m-1] + dUdx[n-2,m-1]
                dUdy[n,m] = n*U[n-1,n-m-1]-n*U[n-1,n-m]+dUdy[n-2,m-1]

            if n%2==1 and m==(n-1)//2+1:
                U[n,m] = xx*U[n-1,m] + yy*U[n-1,n-1-m] + xx*U[n-1, m-1] - U[n-2,m-1]
                dUdx[n,m] = n*U[n-1,m]+n*U[n-1,m-1]+dUdx[n-2,m-1]
                dUdy[n,m] = n*U[n-1,n-m-1] + dUdy[n-2,m-1]

            if n%2==0 and m==n//2:
                U[n,m] = 2*xx*U[n-1,m]+2*yy*U[n-1,m-1] - U[n-2,m-1]
                dUdx[n,m] = 2*n*U[n-1,m]+dUdx[n-2,m-1]
                dUdy[n,m] = 2*n*U[n-1,n-m-1]+dUdy[n-2,m-1]


    U, dUdx, dUdy = keep_zernikes(n_zernike_row, U, dUdx, dUdy)

    size = xx.shape
    dUdy /= (size[0] - 1)/2
    dUdx /= (size[1] - 1)/2

    return(U, dUdx, dUdy)


def find_row(number):
    row = 0
    while True:
        row_sum = (row * (row + 1)) // 2
        next_row_sum = ((row + 1) * (row + 2)) // 2
        if number >= row_sum and number < next_row_sum:
            return row
        row += 1
    

def keep_zernikes(n_zernike_row, zer, dzerdx, dzerdy):
    '''
    keep the actual ones
    '''

    true_zer = []
    true_dzerdx = []
    true_dzerdy = []

    for n in range(n_zernike_row):
        for m in range(n+1):
            true_zer.append(zer[n,m])
            true_dzerdx.append(dzerdx[n,m])
            true_dzerdy.append(dzerdy[n,m])

    true_zer = torch.stack(true_zer)
    true_dzerdx = torch.stack(true_dzerdx)
    true_dzerdy = torch.stack(true_dzerdy)
    
    return true_zer, true_dzerdx, true_dzerdy