import numpy as np

def fft2(img):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

def ifft2(img):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img)))