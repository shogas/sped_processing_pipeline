import numpy as np
from pyxem import ElectronDiffraction

from utils.preprocess import cepstrum
from utils.decomposition import decompose_nmf

def factorize(diffraction_patterns, parameters):
    dps = ElectronDiffraction(diffraction_patterns)
    dps.map(cepstrum, inplace=True, show_progressbar=False)
    factor_count = 2
    decompose_nmf(dps, factor_count)
    factors = dps.get_decomposition_factors().data
    loadings = dps.get_decomposition_loadings().data
    # TODO(simonhog): Return factors from highest index with highest loading to get real-space values
    return (factors, loadings), 'decomposition'

