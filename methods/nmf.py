import numpy as np
from pyxem import ElectronDiffraction

""" Factorizes the given diffraction patterns using the NMF implementation from
pyxem, returning factors and loadings. """

def process(diffraction_patterns, parameters):
    """Factorize diffraction patterns into the product of factors and loadings.

    Paramters
    ---------
    diffraction_patterns : numpy.ndarray
        4D numpy array containing the diffraction patterns
    parameters : dict
        Dictionary of parameters:
        'phase_count' : int
            Number of components to factorize into

    Returns
    -------
    results : (factors, loadings)
        Tuple of factors and corresponding loadings
    result_type : string
        decomposition
    """
    # Load data as a pyxem.ElectronDiffraction object
    dp = ElectronDiffraction(diffraction_patterns)

    # Do the actual factorization using decomposition function from pyxem
    # (which uses HyperSpy and sklearn)
    dp.decomposition(
            normalize_poissonian_noise=True,
            algorithm='nmf',
            output_dimension=parameters['phase_count'])

    # Read the results
    factors = dp.get_decomposition_factors().data
    loadings = dp.get_decomposition_loadings().data

    # Factorization is only unique to a constant factor.
    # Scale so that each loading has a maximum value of 1.
    scaling = loadings.max(axis=(1, 2))  # Maximum in each component
    factors *= scaling[:, np.newaxis, np.newaxis]
    loadings *= np.reciprocal(scaling)[:, np.newaxis, np.newaxis]

    return (factors, loadings), 'decomposition'
