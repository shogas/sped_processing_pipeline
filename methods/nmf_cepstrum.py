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
    # TODO(simonhog): Return factors from highest index with highest loading to get real-space values
    return (factors, loadings), 'decomposition'

