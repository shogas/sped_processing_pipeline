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


