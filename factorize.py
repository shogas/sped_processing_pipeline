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


def generate_test_linear_noiseless(parameters):
    source_a = matplotimg.imread(parameters['source_a_file'])[:, :, 0]
    source_b = matplotimg.imread(parameters['source_b_file'])[:, :, 0]
    factors = np.stack((source_a, source_b))

    width = int(parameters['sample_count_width'])
    height = int(parameters['sample_count_height'])
    loadings = np.empty((2, height, width))
    one_third = width // 3
    for y in range(height):
        for x in range(one_third):
            loadings[0, y, x] = 1.0
            loadings[1, y, x] = 0.0
        for x in range(one_third, 2*one_third):
            loadings[0, y, x] = 1 - (x - one_third) / one_third
            loadings[1, y, x] = 1 - loadings[0, y, x]
        for x in range(2*one_third, width):
            loadings[0, y, x] = 0.0
            loadings[1, y, x] = 1.0

    return factors, loadings


def factorizer_debug(diffraction_patterns):
    dps = pyxem.ElectronDiffraction(diffraction_patterns)
    dps.plot()
    plt.show()


def decompose_nmf(diffraction_pattern, factor_count):
    return diffraction_pattern.decomposition(
            True,
            algorithm='nmf',
            output_dimension=factor_count)


def factorizer_nmf(diffraction_patterns):
    dps = pyxem.ElectronDiffraction(diffraction_patterns)

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


def factorizer_cepstrum_nmf(diffraction_patterns):
    dps = pyxem.ElectronDiffraction(diffraction_patterns)
    dps.map(cepstrum, inplace=True, show_progressbar=False)
    factor_count = 2
    decompose_nmf(dps, factor_count)
    factors = dps.get_decomposition_factors().data
    loadings = dps.get_decomposition_loadings().data
    # TODO(simonhog): Return factors from highest index with highest loading to get real-space values
    return factors, loadings


def save_decomposition(output_dir, method_name, factors, loadings):
    for i in range(factors.shape[0]):
        matplotimg.imsave(os.path.join(output_dir, '{}_factors_{}.tiff').format(method_name, i), factors[i])
    for i in range(loadings.shape[0]):
        matplotimg.imsave(os.path.join(output_dir, '{}_loadings_{}.tiff').format(method_name, i), loadings[i])


def main(parameter_file):
    parameters = parameters_parse(parameter_file)

    output_dir = parameters['output_dir'] if 'output_dir' in parameters else ''
    output_dir = os.path.join(output_dir, 'run_{}_{}'.format(parameters['shortname'], parameters['__date_string']))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    methods = {
        # 'debug': factorizer_debug,
        'nmf': factorizer_nmf,
        'cepstrum_nmf': factorizer_cepstrum_nmf
    }

    ground_truth_factors, ground_truth_loadings = generate_test_linear_noiseless(parameters)

    # TODO(simonhog): numpy probably has a way of doing this without the reshape
    factor_count, pattern_width, pattern_height = ground_truth_factors.shape
    factor_count, sample_width, sample_height = ground_truth_loadings.shape
    factors = ground_truth_factors.reshape((factor_count, -1))
    loadings = ground_truth_loadings.reshape((factor_count, -1))
    diffraction_patterns = np.matmul(loadings.T, factors)
    diffraction_patterns = diffraction_patterns.reshape((sample_width, sample_height, pattern_width, pattern_height))

    for name, factorizer in methods.items():
        print('Running {}'.format(name))
        start_time = time.perf_counter()

        factors, loadings = factorizer(diffraction_patterns.copy())

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print('    Elapsed: {}'.format(elapsed_time))
        parameters['__elapsed_time_{}'.format(name)] = elapsed_time
        save_decomposition(output_dir, name, factors, loadings)

    save_decomposition(output_dir, 'ground_truth', ground_truth_factors, ground_truth_loadings)
    parameters_save(parameters, output_dir)


if __name__ == '__main__':
    main(sys.argv[1])
