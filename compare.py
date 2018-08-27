import os
import sys
import time

import numpy as np
import pyxem

import matplotlib
import matplotlib.image as matplotimg

from parameters import parameters_parse, parameters_save

# TODO(simonhog): Temporary while testing. Final script should not rely on
# plotting
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def linear_blend(source_a, source_b, sample_width, sample_height):
    """ Linear blend diffraction pattern sample

        Blend of two sources in width direction, where the first third is
        entirely source a, the last third entirely source b and the middle is a
        linear blend. All samples in height direction are equal.

        source_a: Array of values representing one source
        source_b: Array of values representing the other source
        sample_width:  Number of samples wide
        sample_height: Number of samples high

        Result: Generator yielding the specified values, iterating along width,
        then height.
    """

    one_third = sample_width // 3
    for y in range(sample_height):
        for i in range(one_third):
            yield source_a
        for i in range(1, one_third + 1):
            t = i / one_third
            yield (1-t)*source_a + t*source_b
        for i in range(sample_width - 2*one_third):
            yield source_b


def run_noiseless(parameters, factorizer):
    source_a = matplotimg.imread(parameters['source_a_file'])[:, :, 0]
    source_b = matplotimg.imread(parameters['source_b_file'])[:, :, 0]
    sample_width = int(parameters['sample_count_width'])
    sample_height = int(parameters['sample_count_height'])
    pattern_width, pattern_height = source_a.shape
    diffraction_patterns = linear_blend(
            source_a, source_b,
            sample_width, sample_height)
    return factorizer(diffraction_patterns, sample_width, sample_height, pattern_width, pattern_height)


def factorizer_debug(diffraction_patterns):
    index = 4
    for i in range(index):
        next(diffraction_patterns)
    test_pattern = next(diffraction_patterns)
    print(test_pattern.shape)
    plt.imshow(test_pattern, cmap='gray')
    plt.show()


def pattern_generator_to_signal(diffraction_patterns, sample_width, sample_height, pattern_width, pattern_height):
    dp_array = np.empty((sample_width * sample_height, pattern_width, pattern_height))
    for i, dp in enumerate(diffraction_patterns):
        dp_array[i] = dp
    dp_array = dp_array.reshape((sample_height, sample_width, pattern_width, pattern_height))
    return pyxem.ElectronDiffraction(dp_array)


def decompose_nmf(diffraction_pattern, factor_count):
    return diffraction_pattern.decomposition(
            True,
            algorithm='nmf',
            output_dimension=factor_count)


def factorizer_nmf(diffraction_patterns, sample_width, sample_height, pattern_width, pattern_height):
    dps = pattern_generator_to_signal(diffraction_patterns, sample_width, sample_height, pattern_width, pattern_height)
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


def cepstrum(z):
    z = np.fft.fft2(z)
    z = z**2
    z = np.log(1 + np.abs(z))
    z = np.fft.ifft2(z)
    z = np.fft.fftshift(z)
    z = np.abs(z)
    z = z**2
    return z


def factorizer_cepstrum_nmf(diffraction_patterns, sample_width, sample_height, pattern_width, pattern_height):
    dps = pattern_generator_to_signal(diffraction_patterns, sample_width, sample_height, pattern_width, pattern_height)
    dps.map(cepstrum, inplace=True)
    factor_count = 2
    decompose_nmf(dps, factor_count)
    factors = dps.get_decomposition_factors().data
    loadings = dps.get_decomposition_loadings().data
    return factors, loadings


def save_decomposition(output_dir, method_name, factors, loadings):
    for i in range(factors.shape[0]):
        matplotimg.imsave(os.path.join(output_dir, '{}_factors_{}.tiff').format(method_name, i), factors[i])
    for i in range(loadings.shape[0]):
        matplotimg.imsave(os.path.join(output_dir, '{}_loadings_{}.tiff').format(method_name, i), loadings[i])


def main(parameter_file):
    run_parameters = parameters_parse(parameter_file)

    output_dir = run_parameters['output_dir'] if 'output_dir' in run_parameters else ''
    output_dir = os.path.join(output_dir, 'run_{}_{}'.format(run_parameters['shortname'], run_parameters['__date_string']))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    methods = {
        'nmf': factorizer_nmf,
        'cepstrum_nmf': factorizer_cepstrum_nmf
    }

    for name, factorizer in methods.items():
        print('Running {}'.format(name))
        start_time = time.perf_counter()

        # TODO(simonhog): Check cost of generating. Better to cache, possibly save to disk and read chunked
        factors, loadings = run_noiseless(run_parameters, factorizer)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print('    Elapsed: {}'.format(elapsed_time))
        run_parameters['__elapsed_time_{}'.format(name)] = elapsed_time
        save_decomposition(output_dir, name, factors, loadings)

    parameters_save(run_parameters, output_dir)


if __name__ == '__main__':
    main(sys.argv[1])
