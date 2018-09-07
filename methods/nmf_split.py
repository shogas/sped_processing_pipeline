import numpy as np
from pyxem import ElectronDiffraction

import matplotlib.pyplot as plt
import matplotlib.image as matplotimg

from .nmf import decompose_nmf

def classify(pattern_library, image):
    # TODO(simonhog): This is just for testing, use proper template matching
    threshold = 0.001
    is_factor = np.max(image) - np.min(image) > threshold
    if is_factor:
        image_norm = image/np.max(image)
        differences = np.empty(len(pattern_library))
        for pattern_index, pattern in enumerate(pattern_library):
            differences[pattern_index] = np.sum(np.abs(image_norm - pattern))
        return np.argmin(differences)
    else:
        return None


def factorize(diffraction_patterns, parameters):
    """ Factorize the diffraction patterns using NMF on slices.

    The diffraction patterns are divided into slices, and each of them are then
    decomposed using NMF. The resulting factors matched against a library of
    patterns and then scaled to also match the maximum intensity. This same
    scaling is used to scale the loading, using and correcting for the fact
    that a factorization X = FL only is unique to a factor s, where F -> F' =
    sF and L -> L' = L/s.

    NOTE: Current implementation of pattern matching is temporary.
    """

    # TODO(simonhog): Temporary mock
    pattern_library = [
        matplotimg.imread(parameters['source_a_file'])[:, :, 0],
        matplotimg.imread(parameters['source_b_file'])[:, :, 0]
    ]

    full_width = diffraction_patterns.shape[1]
    full_height = diffraction_patterns.shape[0]
    split_width = int(parameters['split_width']) if 'split_width' in parameters else full_width
    split_height = int(parameters['split_height']) if 'split_height' in parameters else full_height
    factor_count = 2  # TODO(simonhog): Automatic/parameterized

    loadings = np.empty((factor_count, diffraction_patterns.shape[0], diffraction_patterns.shape[1]))
    for split_start_y in range(0, full_height, split_height):
        split_end_y = min(split_start_y + split_height, full_height)
        slice_y = slice(split_start_y, split_end_y)
        for split_start_x in range(0, full_width, split_width):
            split_end_x = min(split_start_x + split_width, full_width)
            slice_x = slice(split_start_x, split_end_x)
            el_diff = ElectronDiffraction(diffraction_patterns[slice_y, slice_x])
            decompose_nmf(el_diff, factor_count)

            # TODO(simonhog): Use template-matching instead
            new_factors = el_diff.get_decomposition_factors().data
            loading_mappings = { factor_index: classify(pattern_library, new_factors[factor_index])
                                    for factor_index in range(factor_count) }
            loading_data = el_diff.get_decomposition_loadings().data
            for new_factor_index, factor_index in loading_mappings.items():
                if factor_index is not None:
                    scaling = np.max(new_factors[new_factor_index]) / np.max(pattern_library[factor_index])
                    loadings[factor_index, slice_y, slice_x] = scaling * loading_data[new_factor_index]

            total_loading = np.sum(loadings[:, slice_y, slice_x], axis=0)
            for factor_index in range(factor_count):
                loadings[factor_index, slice_y, slice_x] /= total_loading

    factors = pattern_library  # TODO(simonhog): Only used patterns
    return factors, loadings

