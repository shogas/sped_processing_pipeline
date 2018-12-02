import numpy as np

def cepstrum(z):
    z = np.fft.fft2(z)
    z = z**2
    z = np.log1p(np.abs(z))
    z = np.fft.ifft2(z)
    z = np.fft.fftshift(z)
    z = np.abs(z)
    z = z**2
    return z


def cepstrum_power(z):
    z = np.fft.fft2(z)
    z = z * np.conj(z)
    np.seterr(divide='ignore')
    z = np.log(z)
    np.seterr(divide='warn')
    z = np.fft.ifft2(z)
    z = np.fft.fftshift(z)
    z = z * np.conj(z)
    return z.real


def cepstrum_complex(z):
    z = np.fft.fft2(z)
    np.seterr(divide='ignore')
    z = np.log(z)
    np.seterr(divide='warn')
    z = np.fft.ifft2(z)
    z = np.fft.fftshift(z)
    return z


def cepstrum_real(z):
    z = np.fft.fft2(z)
    z = np.abs(z)
    np.seterr(divide='ignore')
    z = np.log(z)
    np.seterr(divide='warn')
    z = np.fft.ifft2(z)
    z = np.fft.fftshift(z)
    return z
