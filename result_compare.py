import os
import sys

import numpy as np
import matplotlib.image as matplotimg

from parameters import parameters_parse


def load_decomposition_files(dir, method, file_type, factor_count):
    file_datas = []
    for file_index in range(factor_count):
        filename = os.path.join(dir, '{}_{}_{}.tiff'.format(method, file_type, file_index))
        if os.path.exists(filename):
            file_datas.append(matplotimg.imread(filename)[:, :, 0])
        else:
            print('[WARN]: Missing file {}'.format(filename))
    return np.stack(file_datas)


def load_decompositions(dir, methods):
    # TODO(simonhog): Different methods don't necessarily yield the same number of factors
    factor_count = 2
    decompositions = {}
    for method in methods:
        decompositions[method] = {}
        decompositions[method]['factors']  = np.stack(load_decomposition_files(dir, method, 'factors', factor_count))
        decompositions[method]['loadings'] = np.stack(load_decomposition_files(dir, method, 'loadings', factor_count))
    return decompositions


def run_comparisons(result_directory):
    parameters = parameters_parse(os.path.join(result_directory, 'metadata.txt'))

    methods = ['ground_truth', 'nmf', 'cepstrum_nmf']
    decompositions = load_decompositions(result_directory, methods)

    ground_truth = decompositions['ground_truth']
    del decompositions['ground_truth']


if __name__ == '__main__':
    run_comparisons(sys.argv[1])
