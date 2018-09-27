from pyxem import ElectronDiffraction

from utils.decomposition import decompose_nmf
""" Factorizes the given diffraction patterns using the NMF implementation from
pyxem, returning factors and loadings. """

def factorize(diffraction_patterns, parameters):
    dps = ElectronDiffraction(diffraction_patterns)

    # dps.decomposition(True, algorithm='svd')
    # dps.plot_explained_variance_ratio()
    # TODO(simonhog): Automate getting number of factors
    factor_count = 2

    decompose_nmf(dps, factor_count)

    # dps.plot_decomposition_results()
    # plt.show()
    factors = dps.get_decomposition_factors().data
    loadings = dps.get_decomposition_loadings().data
    return factors, loadings

