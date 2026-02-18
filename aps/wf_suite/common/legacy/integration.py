import numpy as np

'''
    the 2D integration method here. fourier_integreation1 and 2 and frankotchellappa method
'''
def fourier_integration2DShift(dpc_x=None, del_x=1,
                               dpc_y=None, del_y=1):

    '''
    This function is the direct use of the CONTINOUS formulation of
    Frankot-Chellappa, eq 21 in the article:

    T. Frankot and R. Chellappa
        A Method for Enforcing Integrability in Shape from Shading Algorithms,
        IEEE Transactions On Pattern Analysis And Machine Intelligence, Vol 10,
        No 4, Jul 1988

    In addition, it uses the CONTINOUS shift property to avoid singularities
    at zero frequencies
    input:
        dpc_x:              the differential phase along x
        del_x:              pixel size along x
        dpc_y:              the differential phase along y 
        del_y:              pixel size along y         
    output:
        phi:                phase calculated from the dpc
    '''
    dpc_x = dpc_x / del_x
    dpc_y = dpc_y / del_y

    dim = dpc_x.shape
    x_axis = del_x * (np.arange(dim[1]) - round(dim[1]/2))
    y_axis = del_y * (np.arange(dim[0]) - round(dim[0]/2))

    xx, yy = np.meshgrid(x_axis, y_axis)

    dk_x = 2*np.pi / (dim[1]*del_x)
    dk_y = 2*np.pi / (dim[0]*del_y)

    kx = dk_x * (np.arange(dim[1]) - round(dim[1]/2))
    ky = dk_y * (np.arange(dim[0]) - round(dim[0]/2))

    fy, fx = np.meshgrid(ky/2/np.pi, kx/2/np.pi, indexing='ij')

    fft2 = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
    ifft2 = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))
    # fftshift = lambda x: np.fft.fftshift(x)
    # ifftshift = lambda x: np.fft.ifftshift(x)

    # fx, fy = np.meshgrid(np.fft.fftfreq(dpc_x.shape[1], delx),
    #                      np.fft.fftfreq(dpc_x.shape[0], dely))

    # xx, yy = wpu.realcoordmatrix(dpc_x.shape[1], delx,
    #                              dpc_x.shape[0], dely)

    fo_x = np.abs(fx[0,1]*0.33) # shift fx value
    fo_y = np.abs(fy[1,0]*0.33) # shift fy value


    phaseShift = np.exp(2*np.pi*1j*(fo_x*xx + fo_y*yy))  # exp factor for shift

    mult_factor = 1/(2*np.pi*1j)/(fx - fo_x - 1j*fy + 1j*fo_y )


    bigGprime = fft2((dpc_x - 1j*dpc_y)*phaseShift)
    bigG = bigGprime*mult_factor

    phi = ifft2(bigG) / phaseShift

    phi = np.real(phi) - np.min(np.real(phi))  # since the integral have and undefined constant,
                         # here it is applied an arbritary offset


    return phi


def gradient_nabla(phi, method='FFT'):
    '''
        use fourier transfrom to calculate the displacement from the phase
        here phase is in a unit of [1/(k_wave)/p_x^2*z]
    '''
    if method == 'FFT':
        fft2 = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
        ifft2 = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))
        fftshift = lambda x: np.fft.fftshift(x)

        NN, MM = phi.shape

        wx, wy = np.meshgrid(np.fft.fftfreq(MM)*2*np.pi,
                            np.fft.fftfreq(NN)*2*np.pi, indexing='xy')

        wx = fftshift(wx)
        wy = fftshift(wy)

        displace_x = np.real(ifft2(1j*wx*fft2(phi)))
        displace_y = np.real(ifft2(1j*wy*fft2(phi)))
    elif method == 'diff':
        displace_x = np.gradient(phi, axis=1)
        displace_y = np.gradient(phi, axis=0)
    else:
        print('wrong method for subpixel shift, should be FFT or interp')
        import sys
        sys.exit()

    return [displace_x, displace_y]


def frankotchellappa(dpc_x,dpc_y):
    '''
        Frankt-Chellappa Algrotihm
        input:
            dpc_x:              the differential phase along x
            dpc_y:              the differential phase along y       
        output:
            phi:                phase calculated from the dpc
    '''
    fft2 = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
    ifft2 = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))
    fftshift = lambda x: np.fft.fftshift(x)

    NN, MM = dpc_x.shape
    
    wx, wy = np.meshgrid(np.fft.fftfreq(MM)*2*np.pi,
                         np.fft.fftfreq(NN)*2*np.pi, indexing='xy')
    wx = fftshift(wx)
    wy = fftshift(wy)
    numerator = -1j*wx*fft2(dpc_x) -1j*wy*fft2(dpc_y)
    # here use the numpy.fmax method to eliminate the zero point of the division
    denominator = np.fmax((wx)**2 + (wy)**2, np.finfo(float).eps)

    div = numerator/denominator


    phi = np.real(ifft2(div))

    phi -= np.mean(np.real(phi))

    return phi

def frankotchellappa_ord2(dpc_x,dpc_y):
    '''
        Frankt-Chellappa Algrotihm
        input:
            dpc_x:              the differential phase along x
            dpc_y:              the differential phase along y       
        output:
            phi:                phase calculated from the dpc
    '''
    fft2 = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
    ifft2 = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))
    fftshift = lambda x: np.fft.fftshift(x)

    NN, MM = dpc_x.shape

    wx, wy = np.meshgrid(np.fft.fftfreq(MM)*2*np.pi,
                         np.fft.fftfreq(NN)*2*np.pi, indexing='xy')
    wx = fftshift(wx)
    wy = fftshift(wy)
    numerator = fft2(dpc_x) + fft2(dpc_y)
    # here use the numpy.fmax method to eliminate the zero point of the division
    denominator = np.fmax((wx)**2 + (wy)**2, np.finfo(float).eps)

    div = numerator/denominator


    phi = np.real(ifft2(div))

    phi -= np.mean(np.real(phi))

    return phi



def frankotchellappa_1D(dpc, axis=0):
    # do frankotchellappa integration along specific axis
    fftshift = lambda x: np.fft.fftshift(x, axes=axis)
    ifftshift = lambda x: np.fft.ifftshift(x, axes=axis)
    fft = lambda x: fftshift(np.fft.fft(ifftshift(x), axis=axis))
    ifft = lambda x: fftshift(np.fft.ifft(ifftshift(x), axis=axis))

    NN, MM = dpc.shape

    wx, wy = np.meshgrid(np.fft.fftfreq(MM) * 2 * np.pi,
                         np.fft.fftfreq(NN) * 2 * np.pi,
                         indexing='xy')
    wx = fftshift(wx)
    wy = fftshift(wy)
    ww = wx if axis==1 else wy

    numerator = -1j * ww * fft(dpc)
    # here use the np.fmax method to eliminate the zero point of the division
    denominator = np.fmax((ww)**2, np.finfo(float).eps)

    div = numerator / denominator

    integ = np.real(ifft(div))

    integ -= np.mean(np.real(integ))

    return integ