import numpy as np

def cepstrum(z):
    z = np.fft.fft2(z)
    z = z**2
    z = np.log(1 + np.abs(z))
    z = np.fft.ifft2(z)
    z = np.fft.fftshift(z)
    z = np.abs(z)
    z = z**2
    return z

