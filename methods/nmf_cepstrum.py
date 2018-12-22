import numpy as np
from pyxem import ElectronDiffraction

from utils.preprocess import cepstrum_power
from utils.decomposition import decompose_nmf

def process(diffraction_patterns, parameters):
    phase_count = parameters['phase_count']

    dps = ElectronDiffraction(diffraction_patterns)
    dps_cepstrum = dps.map(cepstrum_power, inplace=False, show_progressbar=False)
    decompose_nmf(dps_cepstrum, phase_count)

    factors = dps_cepstrum.get_decomposition_factors().data
    loadings = dps_cepstrum.get_decomposition_loadings().data

    # Factorization is only unique to a constant factor.
    # Scale so that each loading has a maximum value of 1.
    scaling = loadings.max(axis=(1, 2))  # Maximum in each component
    factors *= scaling[:, np.newaxis, np.newaxis]
    loadings *= np.reciprocal(scaling)[:, np.newaxis, np.newaxis]

    return (factors, loadings), 'decomposition'

