import os
import sys

import numpy as np
import matplotlib
from PIL import Image

from parameters import parameters_parse

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from common import result_image_file_info


def load_decomposition_files(dir, method, file_type, factor_count):
    file_datas = []
    for file_index in range(factor_count):
        filename = os.path.join(dir, '{}_{}_{}.tiff'.format(method, file_type, file_index))
        if os.path.exists(filename):
            file_datas.append(matplotimg.imread(filename)[:, :, 0])
        else:
            print('[WARN]: Missing file {}'.format(filename))
    return np.stack(file_datas)


def calculate_difference(ground_truth_info, method_info):
    ground_truth_data = np.asarray(Image.open(ground_truth_info['filename']))
    method_data = np.asarray(Image.open(method_info['filename']))

    residuals = ground_truth_data - method_data
    sum_square_residuals = np.sum(residuals**2)
    sum_square_total = np.sum((ground_truth_data - np.mean(ground_truth_data))**2)
    r_squared = 1 - sum_square_residuals / sum_square_total

    return r_squared


def run_comparisons(result_directory):
    parameters = parameters_parse(os.path.join(result_directory, 'metadata.txt'))

    factor_count = 2
    methods = [method.strip() for method in parameters['methods'].split(',')]
    factor_infos = result_image_file_info(result_directory, 'factors')
    loading_infos = result_image_file_info(result_directory, 'loadings')

    ground_truth_method = 'ground_truth'
    if ground_truth_method not in factor_infos:
        ground_truth_method = next(iter(factor_infos.keys()))
        print('No ground truth to compare against, comparing against {}'.format(ground_truth_method))

    def factorization_slice_sort(info):
        return (info['x_start'], info['y_start'])

    ground_truth_factors = sorted(factor_infos[ground_truth_method], key=factorization_slice_sort)
    ground_truth_loadings = sorted(loading_infos[ground_truth_method], key=factorization_slice_sort)
    for (method, factor_infos_for_method), loading_infos_for_method in zip(factor_infos.items(), loading_infos.values()):
        if method == ground_truth_method: continue
        print('Comparing {}'.format(method))
        factors = sorted(factor_infos_for_method, key=factorization_slice_sort)
        loadings = sorted(loading_infos_for_method, key=factorization_slice_sort)
        for ground_truth_factor_info, ground_truth_loading_info, factor_info, loading_info in zip(ground_truth_factors, ground_truth_loadings, factors, loadings):
            factor_difference_rsquare = calculate_difference(ground_truth_factor_info, factor_info)
            loading_difference_rsquare = calculate_difference(ground_truth_loading_info, loading_info)
            print('{:4d},{:4d}:'.format(factor_info['x_start'], factor_info['y_start']))
            print('           R^2 factors:  {}'.format(factor_difference_rsquare))
            print('           R^2 loadings: {}'.format(loading_difference_rsquare))


        # for ground_index in range(factor_count):
            # for method_index in range(factor_count):
                # ground_loadings = ground_truth['loadings'][ground_index] 
                # method_loadings = decompositions[method]['loadings'][method_index]
                # loadings_difference = np.abs(ground_loadings - method_loadings)
                # plt.subplot(3, 1, 1)
                # plt.title('ground {}'.format(ground_index))
                # plt.imshow(ground_loadings, cmap='gray')
                # plt.title('{} {}'.format(method, method_index))
                # plt.subplot(3, 1, 2)
                # plt.imshow(method_loadings, cmap='gray')
                # plt.subplot(3, 1, 3)
                # plt.imshow(loadings_difference, cmap='gray')
                # plt.show()


if __name__ == '__main__':
    run_comparisons(sys.argv[1])
