import torch

def Axicon(xx, yy, x0=0, y0=0, truncate_circle = False):
    """
    The axicon is a conical surface, which gives a wavefront that linearly depends on the radial coordinate.

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
        The Phase
    dUdx : array_like
        The derivative of the Axicon Phase with respect to x.
    dUdy : array_like
        The derivative of the Axicon Phase with respect to y.

    """

    rr = ((xx-x0)**2+(yy-y0)**2)**.5

    U = rr
    dUdy = (yy-y0)/(rr)
    dUdx = (xx-x0)/(rr)

    U -= U.mean()

    if truncate_circle==True:
        rr_nan = ((xx)**2+(yy)**2)**.5
        U[rr_nan>1]=torch.nan
        dUdx[rr_nan>1]=torch.nan
        dUdy[rr_nan>1]=torch.nan

    size = xx.shape
    dUdy /= (size[0] - 1)/2
    dUdx /= (size[1] - 1)/2

    return(U.unsqueeze(0), dUdx.unsqueeze(0), dUdy.unsqueeze(0))