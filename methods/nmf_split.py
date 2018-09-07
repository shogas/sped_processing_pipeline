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

    split_width = int(parameters['split_width'])
    full_width = diffraction_patterns.shape[1]
    factor_count = 2  # TODO(simonhog): Automatic/parameterized
    factors = pattern_library  # TODO(simonhog): Only used patterns
    loadings = np.empty((factor_count, diffraction_patterns.shape[0], diffraction_patterns.shape[1]))
    for split_start in range(0, full_width, split_width):
        split_end = min(split_start + split_width, full_width)
        dps = ElectronDiffraction(diffraction_patterns[:, split_start:split_end])
        decompose_nmf(dps, factor_count)

        # TODO(simonhog): Use template-matching instead
        new_factors = dps.get_decomposition_factors().data
        loading_mappings = { factor_index: classify(pattern_library, new_factors[factor_index]) for factor_index in range(factor_count)}
        loading_data = dps.get_decomposition_loadings().data
        for new_factor_index, factor_index in loading_mappings.items():
            if factor_index is not None:
                loading = loading_data[new_factor_index]  #/total_loading
                scaling = np.max(new_factors[new_factor_index]) / np.max(pattern_library[factor_index])
                loading *= scaling
                loadings[factor_index, :, split_start:split_end] = loading

        total_loading = np.sum(loadings[:, :, split_start:split_end], axis=0)
        for factor_index in range(factor_count):
            loadings[factor_index, :, split_start:split_end] /= total_loading

    plt.subplot(2, 1, 1)
    plt.imshow(loadings[0])
    plt.subplot(2, 1, 2)
    plt.imshow(loadings[1])
    plt.show()
    return factors, loadings

