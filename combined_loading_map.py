import os
import sys

import numpy as np
from PIL import Image

from common import result_image_file_info
from parameters import parameters_parse

def image_difference(image_a, image_b):
    return np.sum((image_a - image_b)**2)


def combine_loading_map(method, factor_infos, loading_infos):
    # print(method)
    # print('Factors')
    # for i in factor_infos: print(i)
    # print('Loadings')
    # for i in loading_infos: print(i)
    # return

    total_width  = max((info['x_stop'] for info in loading_infos))
    total_height = max((info['y_stop'] for info in loading_infos))
    print(total_width, total_height)

    combined_loadings = np.zeros((total_height, total_width, 3))

    colors = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, 0.5],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            ])

    threshold = 50
    known_factors = []
    for factor_info, loading_info in zip(factor_infos, loading_infos):
        factor = np.asarray(Image.open(factor_info['filename']))
        factor = factor/factor.max()
        if len(known_factors) > 0:
            diffs = [image_difference(factor, known_factor) for known_factor in known_factors]
            best_diff_index = np.argmin(diffs)
            best_diff = diffs[best_diff_index]
            if (best_diff < threshold):
                print('Matched phase {} (difference {})'.format(best_diff_index, best_diff))
                factor_index = best_diff_index
            else:
                print('New phase {} (difference {})'.format(best_diff_index, best_diff))
                factor_index = len(known_factors)
                known_factors.append(factor)
        else:
            factor_index = len(known_factors)
            known_factors.append(factor)

        loading = np.asarray(Image.open(loading_info['filename']))
        print('Factor index: {}'.format(factor_index))
        print(colors[factor_index])
        print(factor_info['x_start'], factor_info['x_stop'])
        print(factor_info['y_start'], factor_info['y_stop'])
        print()
        combined_loadings[
                factor_info['y_start']:factor_info['y_stop'],
                factor_info['x_start']:factor_info['x_stop']] += np.outer(loading.ravel(), colors[factor_index]).reshape(loading.shape[0], loading.shape[1], 3)

    combined_loadings *= 255 / combined_loadings.max()
    return Image.fromarray(combined_loadings.astype('uint8'))


def combine_loading_maps(result_directory):
    parameters = parameters_parse(os.path.join(result_directory, 'metadata.txt'))

    methods = [method.strip() for method in parameters['methods'].split(',')]
    factor_infos = result_image_file_info(result_directory, 'factors')
    loading_infos = result_image_file_info(result_directory, 'loadings')
    for (method_name, factor_infos_for_method), loading_infos_for_method in zip(factor_infos.items(), loading_infos.values()):
        combined_loadings = combine_loading_map(method_name, factor_infos_for_method, loading_infos_for_method)
        combined_loadings.save(os.path.join(result_directory, 'loading_map_{}.tiff'.format(method_name)))

if __name__ == '__main__':
    combine_loading_maps(sys.argv[1])
