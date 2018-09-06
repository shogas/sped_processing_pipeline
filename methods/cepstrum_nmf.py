import numpy as np
from pyxem import ElectronDiffraction

from .nmf import decompose_nmf

def cepstrum(z):
    z = np.fft.fft2(z)
    z = z**2
    z = np.log(1 + np.abs(z))
    z = np.fft.ifft2(z)
    z = np.fft.fftshift(z)
    z = np.abs(z)
    z = z**2
    return z


def factorize(diffraction_patterns):
    dps = ElectronDiffraction(diffraction_patterns)
    dps.map(cepstrum, inplace=True, show_progressbar=False)
    factor_count = 2
    decompose_nmf(dps, factor_count)
    factors = dps.get_decomposition_factors().data
    loadings = dps.get_decomposition_loadings().data
    # TODO(simonhog): Return factors from highest index with highest loading to get real-space values
    return factors, loadings

