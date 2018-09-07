from pyxem import ElectronDiffraction

""" Factorizes the given diffraction patterns using the NMF implementation from
pyxem, returning factors and loadings. """

def decompose_nmf(diffraction_pattern, factor_count):
    """ Decompose the given diffraction patterns using NMF.

    Results are stored in the ElectronDiffraction object.

    Args:
        diffraction_pattern: diffraction patterns of type
            pyxem.ElectronDiffraction
        factor_count: number of factors to decompose into
    """
    diffraction_pattern.decomposition(
            True,
            algorithm='nmf',
            output_dimension=factor_count)


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

