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
        print(loading_mappings)

        # dps.plot_decomposition_results()
        loading_data = dps.get_decomposition_loadings().data
        total_loading = np.sum(loading_data[new_factor_index] for new_factor_index, factor_index in loading_mappings.items() if factor_index is not None)
        for new_factor_index, factor_index in loading_mappings.items():
            if factor_index is not None:
                print('Moving {} -> {}'.format(new_factor_index, factor_index))
                loading = loading_data[factor_index]  #/total_loading
                plt.imshow(new_factors[new_factor_index])
                plt.show()
                loadings[new_factor_index, :, split_start:split_end] = loading

        print()

    plt.subplot(2, 1, 1)
    plt.imshow(loadings[0])
    plt.subplot(2, 1, 2)
    plt.imshow(loadings[1])
    plt.show()
    return factors, loadings





# new_factors = dps.get_decomposition_factors().data
# for new_factor_index in range(factor_count):
#     threshold = 0.001
#     ptp = np.max(new_factors[new_factor_index]) - np.min(new_factors[new_factor_index])
#     is_factor = ptp > threshold
#     print('New {} is factor: {} ({})'.format(new_factor_index, is_factor, ptp))
#     if is_factor:
#         differences = np.empty(factor_count)
#         for factor_index in range(factor_count):
#             if factor_found[factor_index]:
#                 # differences[factor_index] = np.linalg.norm(factors[factor_index] - new_factors[new_factor_index])
#                 fac = factors[factor_index]/np.max(factors[factor_index])
#                 fac_new = new_factors[new_factor_index]/np.max(factors[factor_index])
#                 diff = np.abs(fac - fac_new)
#                 differences[factor_index] = np.sum(diff)
#                 print('Difference f({}), n({}): {}'.format(factor_index, new_factor_index, differences[factor_index]))
#                 plt.subplot(2, 2, 1)
#                 plt.title('f {}'.format(factor_index))
#                 plt.imshow(fac)
#                 plt.subplot(2, 2, 2)
#                 plt.title('n {}'.format(new_factor_index))
#                 plt.imshow(fac_new)
#                 plt.subplot(2, 2, 3)
#                 plt.title('diff')
#                 plt.imshow(diff)
#                 plt.show()
#             else:
#                 differences[factor_index] = 1.0
#         print(differences)
#         loading_mappings[new_factor_index] = np.argmin(differences)

