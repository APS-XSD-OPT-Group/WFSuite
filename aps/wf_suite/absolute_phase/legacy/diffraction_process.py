'''
This is the python version code for diffraction
                dxy             the pixel pitch of the object
                z               the distance of the propagation
                wavelength      the wave length
                X,Y             meshgrid of coordinate
                data            input object
                diff_method     calculation method:
                                QPF: quadratic phase fresnel diffraction
                                IR:  convolution transfer methord (for far field)
                                TF: spectral transfer method    (for near field)
                                RS: Rayleigh-Sommerfield method
                                default: use IR(far) and TF(near)
'''
import numpy as np
import sys
import scipy.constants as sc
import scipy.ndimage as sf
import scipy.interpolate as sfit
from skimage.restoration import unwrap_phase


def diffraction_prop(data, dxy, z, wavelength, diff_method='default'):
    '''
    This is the python version code for diffraction
        dxy             the pixel pitch of the object
        z               the distance of the propagation
        wavelength      the wave length
        X,Y             meshgrid of coordinate
        data            input object
        diff_method     calculation method:
                        QPF: quadratic phase fresnel diffraction
                        IR:  convolution transfer methord (for far field)
                        TF: spectral transfer method    (for near field)
                        RS: Rayleigh-Sommerfield method
                        GS: phase gradient shift, approximation of near field diffraction
                        default: use IR(far) and TF(near)
        if donot specify, the method will be determined by dxy and L(source length)
    '''
    if diff_method == 'default':
        # the method is not defined
        # the array size
        M = data.shape[0]
        # the source plane size
        L = M * dxy
        if dxy > wavelength * z / L:
            # use TF method for near field
            diff, L_out = prop_TF(data, dxy, z, wavelength)
        else:
            # use IR method for far field
            diff, L_out = prop_IR(data, dxy, z, wavelength)

    elif diff_method == 'QPF':
        # the method is defined
        # use QPF method
        diff, L_out = prop_QPF(data, dxy, z, wavelength)
    elif diff_method == 'IR':
        # use IR method
        diff, L_out = prop_IR(data, dxy, z, wavelength)
    elif diff_method == 'TF':
        # use TF method
        diff, L_out = prop_TF(data, dxy, z, wavelength)
    elif diff_method == 'RS':
        # use RS method
        diff, L_out = prop_RS(data, dxy, z, wavelength)
    elif diff_method == 'GS':
        # use phase gradient shift method
        diff, L_out = prop_GS(data, dxy, z, wavelength)
    elif diff_method == 'GeoFlow':
        # use phase gradient shift method
        diff, L_out = prop_GeoFlow(np.abs(data)**2, np.angle(data), dxy, z, wavelength)
    else:
        sys.exit('Error: no such diffraction method; must be TF, IR, RS, QPF')

    return diff, L_out


def prop_GeoFlow(I_ref, phi, dxy, z, wavelength):
    '''
        here use the Geometric flow method to calculate the img after propagation
        input:
            I_ref:                  the ref intensity, I_ref=abs(A_ref)**2
            phi:                    the phase
            dxy:                    pixel size
            z:                      the propagation distance
            wavelength:             the wavelength
        output:
            I_img:                  the intensity after propagation
        Formula:    I_ref - I_img = diff_T(I_ref * D_T(x, y))
                    D_T(x, y) is the displacement along x and y axis
    '''
    dim = I_ref.shape
    k_wave = 2 * np.pi / wavelength

    x_axis = dxy * (np.arange(dim[1]) - round(dim[1]/2))
    y_axis = dxy * (np.arange(dim[0]) - round(dim[0]/2))
    dk_x = 2*np.pi / (dim[1]*dxy)
    dk_y = 2*np.pi / (dim[0]*dxy)
    # dk_x = 1 / (dim[1]*dxy)
    # dk_y = 1 / (dim[0]*dxy)

    kx = dk_x * (np.arange(dim[1]) - round(dim[1]/2))
    ky = dk_y * (np.arange(dim[0]) - round(dim[0]/2))

    k_YY, k_XX = np.meshgrid(ky, kx, indexing='ij')
    fft2 = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
    ifft2 = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))
    '''
        to calculate the diff_T:
            diff_T = 1j * F^(-1){(kx, ky) * F{}}
    '''
    # axis=1 direction deviation
    # nabla_x_kern = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    # # axis=0 direction deviation
    # nabla_y_kern = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    # # 2nd nabla deviation
    # nabla_2_kern = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # bc = 'reflect'
    # nabla = lambda phi: np.array([sf.correlate(phi, nabla_x_kern, mode=bc), sf.correlate(phi, nabla_y_kern, mode=bc)]) / (dxy)
    # nabla2 = lambda phi: sf.correlate(phi, nabla_2_kern, mode=bc) / dxy**2
    nabla_T = lambda x: 1j * np.array([ifft2(k_YY * fft2(x)), ifft2(k_XX * fft2(x))])
    # nabla_times = lambda x, y: x[0]*y[0] + x[1]*y[1]
    # nabla2 = lambda vec: ifft2(-(k_XX**2 + k_YY**2) * fft2(vec))

    # I_img = I_ref - (z/k_wave * (nabla_times(nabla(I_ref), nabla(phi)) + I_ref * nabla2(phi)))
    # D_T = np.real(diff_T(z / k_wave * phi))
    
    # D_x = np.real(ifft2((1j*k_XX - k_YY) * fft2(phi * z / k_wave)))
    # D_y = np.imag(ifft2((1j*k_XX - k_YY) * fft2(phi * z / k_wave)))
    [D_y, D_x] = np.real(nabla_T(phi * z / k_wave))
    # use the formula to calculate intensity after propagation
    # I_ref - I_img = diff_T(I_ref * D_T(x, y))

    I_img = I_ref - np.real(1j*ifft2(k_XX * fft2(I_ref * D_x) + k_YY * fft2(I_ref * D_y)))

    return I_img, dxy * I_ref.shape[0]


def prop_GS(data, dxy, z , wavelength):
    '''
    Use phase gradient shift method for the diffraction. Assume same x and y lengthss and uniform sampling
        data:        source plane field
        dxy:         source and observation plane pixel size
        wavelength:  wavelength
        z:           propagation distance
        GS method
        dx, dy = -lambda*z/2/pi * nabla(phase)

    '''
    # phase of the data
    L = dxy * data.shape[0]
    phi = unwrap_phase(np.angle(data))
    I_amp = np.abs(data) ** 2
    
    cons = wavelength * z / (2 * np.pi)

    x_axis = range(I_amp.shape[1])
    y_axis = range(I_amp.shape[0])
    YY_axis, XX_axis = np.meshgrid(y_axis, x_axis, indexing='ij')

    wx = np.gradient(phi, axis=1) * cons / dxy**2
    wy = np.gradient(phi, axis=0) * cons / dxy**2
    YY_axis = YY_axis - wy
    XX_axis = XX_axis - wx

    f_amp = sfit.RegularGridInterpolator((y_axis, x_axis), I_amp, 
                                                  bounds_error = False, 
                                                  method = 'nearest', 
                                                  fill_value = np.mean(I_amp))
    f_phase = sfit.RegularGridInterpolator((y_axis, x_axis), phi, 
                                                  bounds_error = False, 
                                                  method = 'nearest', 
                                                  fill_value = np.mean(phi))
    # recently inverted this line
    pts = (np.array([np.ndarray.flatten(YY_axis), np.ndarray.flatten(XX_axis)])
           .transpose())

    I_new = np.reshape(f_amp(pts), I_amp.shape)
    phi_new = np.reshape(f_phase(pts), phi.shape)

    Amp_new = np.sqrt(I_new) * np.exp(1j * phi_new)

    return Amp_new, L



def prop_RS(data, dxy, z, wavelength):
    '''
    Use Rayleigh-Sommerfield method for the diffraction. Assume same x and y lengthss and uniform sampling
        data:        source plane field
        dxy:         source and observation plane pixel size
        wavelength:  wavelength
        z:           propagation distance
        RS method
        u2(x,y)=ifft(fft(u1)*H); H=exp(jkz(1-(lambdafx)^2-(lambdafy)^2)^0.5)

    '''
    # the array size
    M = data.shape[0]
    N = data.shape[1]
    # source plane size
    L_y = dxy * M
    L_x = dxy * N
    # wavenumber
    k = 2 * np.pi / wavelength
    # frequency resolution
    dfx = 1/L_x
    dfy = 1/L_y
    fy = np.arange(-M/2, M/2) * dfy
    fx = np.arange(-N/2, N/2) * dfx
    FX, FY = np.meshgrid(fx, fy)

    L_out = [L_y, L_x]
    if z > 0:
        # transform function
        H = np.exp(1j*k*z*np.sqrt(1-(wavelength*FX)**2-(wavelength*FY)**2))
        # u2(x,y)=ifft(fft(u1)*H)
        diff = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(data)) * np.fft.ifftshift(H)))
    else:
        # transform function
        H = np.exp(1j*k*z*np.sqrt(1-(wavelength*FX)**2-(wavelength*FY)**2))
        # u2(x,y)=ifft(fft(u1)*H)
        diff = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(data)) * np.fft.ifftshift(H)))

    return diff, L_out


def prop_QPF(data, dxy, z, wavelength):
    '''
    this method use the quadratic phase method to calculate fresnel diffraction
        data:        source plane field
        dxy:         source and observation plane pixel size
        wavelength:  wavelength
        z:           propagation distance
        QPF:
            U2(x,y) = exp(jkz)/(j*lambda*z)*exp(jk/2z*(x^2+y^2))*int(U1(xx,yy)exp(jk/2z*(xx^2+yy^2))*exp(-jk/z(x*xx+y*yy))dxxdyy)

    '''
    # the array size
    M = data.shape[0]
    N = data.shape[1]
    # source plane size
    L_y = dxy * M
    L_x = dxy * N
    # wavenumber
    k = 2 * np.pi / wavelength
    # spatial resolution
    y = np.arange(-N/2, N/2) * dxy
    x = np.arange(-M/2, M/2) * dxy
    XX, YY = np.meshgrid(x, y)
    # frequency resolution
    dfx = 1/L_x * wavelength * z
    dfy = 1/L_y * wavelength * z
    fy = np.arange(-M/2, M/2) * dfy
    fx = np.arange(-N/2, N/2) * dfx

    FX, FY = np.meshgrid(fx, fy)

    # the out plane size
    L_out = wavelength * z / dxy

    if z > 0:
        pf = np.exp(1j*k*z) * np.exp(1j*k/(2*z)*(FX**2 + FY**2))
        kern = data * np.exp(1j*k/(2*z)*(XX**2 + YY**2))
        cgh = np.fft.fft2(np.fft.ifftshift(kern))
        diff = np.fft.fftshift(cgh * np.fft.ifftshift(pf))
    else:
        pf = np.exp(1j*k*z) * np.exp(1j*k/(2*z)*(XX**2 + YY**2))
        kern = data * np.exp(1j*k/(2*z)*(FX**2 + FY**2))
        cgh = np.fft.ifft2(np.fft.ifftshift(kern))
        diff = np.fft.fftshift(cgh)*pf

    return diff, L_out


def prop_TF(data, dxy, wavelength, z):
    '''
    this method use the transfer function approach to calculate fresnel diffraction
        data:        source plane field
        dxy:         source and observation plane pixel size
        wavelength:  wavelength
        z:           propagation distance
        TF:
            U2(x,y)=ifft(fft(u1)*H); H=exp(jkz)*exp(-j*pi*lambda*z*(fx^2+fy^2))

    '''
    # the array size
    M = data.shape[0]
    N = data.shape[1]
    # source plane size
    L_x = dxy * N
    L_y = dxy * M
    # wavenumber
    k = 2 * np.pi / wavelength
    # frequency resolution
    dfx = 1/L_x
    dfy = 1/L_y
    fy = np.arange(-M/2, M/2) * dfy
    fx = np.arange(-N/2, N/2) * dfx

    FX, FY = np.meshgrid(fx, fy)

    L_out = [L_y, L_x]

    if z > 0:
        # transfer function
        H = np.exp(1j*k*z) * np.exp(-1j*wavelength*z*np.pi*(FX**2 + FY**2))
        diff = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(data)) * np.fft.ifftshift(H)))
    else:
        # transfer function
        H = np.exp(1j*k*z) * np.exp(-1j*wavelength*z*np.pi*(FX**2 + FY**2))
        diff = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(data)) * np.fft.ifftshift(H)))

    return diff, L_out


def prop_IR(data, dxy, wavelength, z):
    '''
    this method use the impulse response approach to calculate fresnel diffraction
        data:        source plane field
        dxy:         source and observation plane pixel size
        wavelength:  wavelength
        z:           propagation distance
        IR:
            U2(x,y)=ifft(fft(u1)*H); H=fft(exp(jkz)/(jlambda*z)*exp(j*k/2z*(x^2+y^2)))

    '''
    # the array size
    M = data.shape[0]
    # source plane size
    L = dxy * M
    # wavenumber
    k = 2 * np.pi / wavelength
    # spatial resolution
    x = np.arange(-M/2, M/2) * dxy
    XX, YY = np.meshgrid(x, x)

    L_out = L

    if z > 0:
        # impule response
        h = 1/(1j*wavelength*z) * np.exp(1j*k/(2*z)*(XX**2 + YY**2))
        # transfer function
        H = np.fft.fft2(np.fft.ifftshift(h)) * dxy**2
        U1 = np.fft.fft2(np.fft.ifftshift(data))
        diff = np.fft.fftshift(np.fft.ifft2(U1 * H))
    else:
        # impule response
        h = 1/(-1j*wavelength*z) * np.exp(1j*k/(-2*z)*(XX**2 + YY**2))
        # transfer function
        H = np.fft.fft2(np.fft.ifftshift(h)) * dxy**2
        U1 = np.fft.fft2(np.fft.ifftshift(data))
        diff = np.fft.fftshift(np.fft.ifft2(U1 / H))

    return diff, L_out


# here is the old propagation function for Fresnel diffraction
def fresnel_propagation(data, dxy, z, wavelength):

    (M, N) = data.shape
    k = 2 * np.pi / wavelength
    # the coordinate grid
    if M % 2 == 0:
        M_grid = np.arange(-M/2, M/2)
    else:
        M_grid = np.arange(-(M-1)/2, (M-1)/2+1)
    lx = M_grid * dxy
    XX, YY = np.meshgrid(lx, lx)

    # the coordinate grid on the output plane
    fc = 1/dxy
    fu = wavelength * z * fc
    lu = M_grid * fu / M
    Fx, Fy = np.meshgrid(lu, lu)

    if z > 0:
        pf = np.exp(1j*k*z) * np.exp(1j*k*(Fx**2 + Fy**2)/2/z)
        kern = data * np.exp(1j*k*(XX**2 + YY**2)/2/z)
        cgh = np.fft.fft2(np.fft.fftshift(kern))
        OUT = np.fft.fftshift(cgh * np.fft.fftshift(pf))
    else:
        pf = np.exp(1j*k*z) * np.exp(1j*k*(XX**2 + YY**2)/2/z)
        cgh = np.fft.ifft2(np.fft.fftshift(data*np.exp(1j*k*(Fx**2+Fy**2)/2/z)))
        OUT = np.fft.fftshift(cgh)*pf

    return OUT



def prop_TF_2d(data, dxy, wavelength, z):
    '''
    this method use the transfer function approach to calculate fresnel diffraction
        data:        source plane field
        dxy:         source and observation plane pixel size
        wavelength:  wavelength
        z:           propagation distance, [zx, zy]
        TF:
            U2(x,y)=ifft(fft(u1)*H); H=exp(jkz)*exp(-j*pi*lambda*z*(fx^2+fy^2))

    '''
    # the array size
    M = data.shape[0]
    N = data.shape[1]
    # source plane size
    L_x = dxy * N
    L_y = dxy * M
    # wavenumber
    k = 2 * np.pi / wavelength
    # frequency resolution
    dfx = 1/L_x
    dfy = 1/L_y
    fy = np.arange(-M/2, M/2) * dfy
    fx = np.arange(-N/2, N/2) * dfx

    FX, FY = np.meshgrid(fx, fy)

    L_out = [L_y, L_x]

    H = np.exp(-1j*wavelength*np.pi*(z[0] * FX**2 + z[1] * FY**2))
    diff = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(data)) * np.fft.ifftshift(H)))

    return diff, L_out

