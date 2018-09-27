import math

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as matplotimg

from pyxem import ElectronDiffraction

from utils.decomposition import decompose_nmf
from utils.template_matching import classify, generate_rotation_list, generate_diffraction_library


phase_index_to_name = {
    0: 'GaAs ZB1',
    1: 'GaAs WZ',
}
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

    diffraction_library = generate_diffraction_library(parameters, phase_index_to_name.values())

    diffraction_pattern_image_size = diffraction_patterns.shape[2]
    full_width = diffraction_patterns.shape[1]
    full_height = diffraction_patterns.shape[0]
    split_width = parameters['split_width'] if 'split_width' in parameters else full_width
    split_height = parameters['split_height'] if 'split_height' in parameters else full_height
    factor_count = 2  # TODO(simonhog): Automatic/parameterized

    # TODO(simonhog): Constants...
    peak_sigma = 0.02
    max_reciprocal = 1

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
            loading_mappings = { new_factor_index: classify(diffraction_library, new_factors[new_factor_index], phase_index_to_name)
                                    for new_factor_index in range(factor_count) }
            loading_data = el_diff.get_decomposition_loadings().data
            for new_factor_index, (factor_index, angles) in loading_mappings.items():
                if factor_index is not None:
                    phase = diffraction_library[phase_index_to_name[factor_index]]
                    # TODO(simonhog): Use actual peak data instead of simulating the image
                    pattern = phase[tuple(angles)]['Sim'].as_signal(diffraction_pattern_image_size, peak_sigma, max_reciprocal)
                    scaling = np.max(new_factors[new_factor_index]) / np.max(pattern.data)
                    loadings[factor_index, slice_y, slice_x] = scaling * loading_data[new_factor_index]

            total_loading = np.sum(loadings[:, slice_y, slice_x], axis=0)
            for factor_index in range(factor_count):
                loadings[factor_index, slice_y, slice_x] /= total_loading

    factors = []
    for phase_name in phase_index_to_name.values():
        phase = diffraction_library[phase_name]
        # TODO(simonhog): Use actual peak data instead of simulating the image
        pattern = phase[next(iter(phase.keys()))]['Sim'].as_signal(diffraction_pattern_image_size, peak_sigma, max_reciprocal)
        factors.append(pattern.data)

    return factors, loadings

