import os
import sys

import numpy as np
from PIL import Image

from tqdm import tqdm as report_progress

import matplotlib
matplotlib.use('Qt5Agg')
import pyxem as pxm
from pyxem.generators.indexation_generator import IndexationGenerator

from common import result_image_file_info
from parameters import parameters_parse
from utils.template_matching import generate_diffraction_library, get_orientation_map

def image_l2_norm(image_a, image_b):
    return np.linalg.norm(image_a - image_b, ord='fro')


def classify_l2_norm(factor, known_factors):
    threshold = 5
    if len(known_factors) > 0:
        diffs = [image_l2_norm(factor, known_factor) for known_factor in known_factors]
        best_diff_index = np.argmin(diffs)
        best_diff = diffs[best_diff_index]
        if (best_diff < threshold):
            factor_index = best_diff_index
            report_progress.write('Matched phase {} (difference {})'.format(factor_index, best_diff))
        else:
            factor_index = len(known_factors)
            known_factors.append(factor)
            report_progress.write('New phase {} (difference {})'.format(factor_index, best_diff))
    else:
        factor_index = len(known_factors)
        known_factors.append(factor)

    return factor_index


def classify_template_match(factor, known_factors):
    # factor = np.exp(factor)
    dp = pxm.ElectronDiffraction([[factor]])
    pattern_indexer = IndexationGenerator(dp, diffraction_library)
    indexation_results = pattern_indexer.correlate(n_largest=4, keys=phase_names, show_progressbar=False)
    crystal_mapping = indexation_results.get_crystallographic_map(show_progressbar=False)
    phases = crystal_mapping.get_phase_map().data.ravel()
    orientations = get_orientation_map(crystal_mapping).data.ravel()
    for phase, orientation in zip(phases, orientations):
        if (phase, orientation) in known_factors:
            factor_index = known_factors.index((phase, orientation))
            report_progress.write('Matched phase {}, {}'.format(phase, orientation))
        else:
            factor_index = len(known_factors)
            known_factors.append((phase, orientation))
            report_progress.write('New phase {}, {}'.format(phase, orientation))

    return factor_index


def combine_loading_map(method, factor_infos, loading_infos):
    total_width  = max((info['x_stop'] for info in loading_infos))
    total_height = max((info['y_stop'] for info in loading_infos))

    classify = classify_l2_norm
    # classify = classify_template_match

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

    known_factors = []
    for factor_info, loading_info in report_progress(zip(factor_infos, loading_infos), total=len(factor_infos)):
        report_progress.write('Tile {}:{}  {}:{} (of {} {})'.format(
            factor_info['x_start'], factor_info['x_stop'],
            factor_info['y_start'], factor_info['y_stop'],
            total_width, total_height))
        factor = np.asarray(Image.open(factor_info['filename']))
        factor = factor/factor.max()

        factor_index = classify(factor, known_factors)
        report_progress.write('Factor index: {}'.format(factor_index))

        loading = np.asarray(Image.open(loading_info['filename']))
        color = colors[factor_index % len(colors)]
        combined_loadings[
                factor_info['y_start']:factor_info['y_stop'],
                factor_info['x_start']:factor_info['x_stop']] += np.outer(loading.ravel(), color).reshape(loading.shape[0], loading.shape[1], 3)

    combined_loadings *= 255 / combined_loadings.max()
    return Image.fromarray(combined_loadings.astype('uint8'))


def combine_loading_maps(result_directory):
    parameters = parameters_parse(os.path.join(result_directory, 'metadata.txt'))

    methods = [
            method.strip() for method in parameters['methods'].split(',')
            if parameters['__save_method_{}'.format(method)] == 'decomposition']
    factor_infos = result_image_file_info(result_directory, 'factors')
    loading_infos = result_image_file_info(result_directory, 'loadings')
    for (method_name, factor_infos_for_method), loading_infos_for_method in zip(factor_infos.items(), loading_infos.values()):
        combined_loadings = combine_loading_map(method_name, factor_infos_for_method, loading_infos_for_method)
        combined_loadings.save(os.path.join(result_directory, 'loading_map_{}.tiff'.format(method_name)))

if __name__ == '__main__':
    # TODO(simonhog): Make these less global. known_factors -> general dictionary for data?
    result_directory = sys.argv[1]
    phase_names = ['ZB', 'WZ']
    diffraction_library = generate_diffraction_library(parameters, phase_names)
    combine_loading_maps(result_directory)
