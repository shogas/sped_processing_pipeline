import glob
import importlib
import os
import sys
import time

import numpy as np

# import matplotlib
import matplotlib.image as matplotimg

from parameters import parameters_parse, parameters_save


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


def save_decomposition(output_dir, method_name, factors, loadings):
    for i in range(factors.shape[0]):
        matplotimg.imsave(os.path.join(output_dir, '{}_factors_{}.tiff').format(method_name, i), factors[i])
    for i in range(loadings.shape[0]):
        matplotimg.imsave(os.path.join(output_dir, '{}_loadings_{}.tiff').format(method_name, i), loadings[i])


def list_available_factorizers():
    """ List all available factorizers in the methods directory """
    print('Available factorizers from methods directory:')
    for module_file in glob.iglob('methods/*.py'):
        factorizer_module = os.path.splitext(os.path.basename(module_file))[0]
        # Look for all python files in the methods subdirectory
        mod = importlib.import_module('methods.{}'.format(factorizer_module))
        factorizer = getattr(mod, 'factorize')
        if factorizer:
            print(' '*4 + factorizer.__name__)


def get_factorizer(name):
    """ Load the factorizer from methods/<name>.py and find the factorize metho

    Args:
        name: name of factorization method, corresponding to the filename methods/<name>.py.
    Returns:
        Factorizer method.
    """
    mod = importlib.import_module('methods.{}'.format(name))
    return getattr(mod, 'factorize')


def run_factorizations(parameters):
    output_dir = parameters['output_dir'] if 'output_dir' in parameters else ''
    output_dir = os.path.join(output_dir, 'run_{}_{}'.format(parameters['shortname'], parameters['__date_string']))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    ground_truth_factors, ground_truth_loadings = generate_test_linear_noiseless(parameters)

    # TODO(simonhog): numpy probably has a way of doing this without the reshape
    factor_count, pattern_width, pattern_height = ground_truth_factors.shape
    factor_count, sample_width, sample_height = ground_truth_loadings.shape
    factors = ground_truth_factors.reshape((factor_count, -1))
    loadings = ground_truth_loadings.reshape((factor_count, -1))
    diffraction_patterns = np.matmul(loadings.T, factors)
    diffraction_patterns = diffraction_patterns.reshape((sample_width, sample_height, pattern_width, pattern_height))

    methods = parameters['methods'].split(',')
    for method_name in methods:
        print('Running factorizer "{}"'.format(method_name))
        start_time = time.perf_counter()

        factorizer = get_factorizer(method_name)

        factors, loadings = factorizer(diffraction_patterns.copy(), parameters)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print('    Elapsed: {}'.format(elapsed_time))
        parameters['__elapsed_time_{}'.format(method_name)] = elapsed_time
        # save_decomposition(output_dir, method_name, factors, loadings)

    save_decomposition(output_dir, 'ground_truth', ground_truth_factors, ground_truth_loadings)
    parameters_save(parameters, output_dir)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        parameters = parameters_parse(sys.argv[1])
    else:
        # TODO(simonhog): Help message
        parameters = {}

    run_factorizations(parameters)
