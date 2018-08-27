import os
import sys

import numpy as np
import matplotlib
import matplotlib.image as matplotimg

from parameters import parameters_parse

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def load_decomposition_files(dir, method, file_type, factor_count):
    file_datas = []
    for file_index in range(factor_count):
        filename = os.path.join(dir, '{}_{}_{}.tiff'.format(method, file_type, file_index))
        if os.path.exists(filename):
            file_datas.append(matplotimg.imread(filename)[:, :, 0])
        else:
            print('[WARN]: Missing file {}'.format(filename))
    return np.stack(file_datas)


def load_decompositions(dir, methods, factor_count):
    # TODO(simonhog): Different methods don't necessarily yield the same number of factors
    decompositions = {}
    for method in methods:
        decompositions[method] = {}
        decompositions[method]['factors']  = np.stack(load_decomposition_files(dir, method, 'factors', factor_count))
        decompositions[method]['loadings'] = np.stack(load_decomposition_files(dir, method, 'loadings', factor_count))
    return decompositions


def run_comparisons(result_directory):
    parameters = parameters_parse(os.path.join(result_directory, 'metadata.txt'))

    factor_count = 2
    methods = ['ground_truth', 'nmf', 'cepstrum_nmf']
    decompositions = load_decompositions(result_directory, methods, factor_count)

    ground_truth = decompositions['ground_truth']
    del decompositions['ground_truth']
    methods = methods[1:]

    for method in methods:
        for ground_index in range(factor_count):
            for method_index in range(factor_count):
                ground_loadings = ground_truth['loadings'][ground_index] 
                method_loadings = decompositions[method]['loadings'][method_index]
                loadings_difference = np.abs(ground_loadings - method_loadings)
                plt.subplot(3, 1, 1)
                plt.title('ground {}'.format(ground_index))
                plt.imshow(ground_loadings, cmap='gray')
                plt.title('{} {}'.format(method, method_index))
                plt.subplot(3, 1, 2)
                plt.imshow(method_loadings, cmap='gray')
                plt.subplot(3, 1, 3)
                plt.imshow(loadings_difference, cmap='gray')
                plt.show()


if __name__ == '__main__':
    run_comparisons(sys.argv[1])
