import glob
import importlib
import os
import re
import sys
import time

import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.image as matplotimg
import matplotlib.pyplot as plt

from PIL import Image
import pickle

import hyperspy.api as hs
import pyxem as pxm
from pyxem.utils.expt_utils import circular_mask
from pyxem.generators.indexation_generator import IndexationGenerator


from parameters import parameters_parse, parameters_save
from common import result_image_file_info


import warnings
# Silence some future warnings and user warnings (float64 -> uint8)
# in skimage when calling remove_background with h-dome (below)
# Should really be fixed elsewhere.
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def generate_test_linear_noiseless(parameters):
    for source_file in ('source_a_file', 'source_b_file'):
        if source_file not in parameters:
            print('No parameter {} given'.format(source_file))
            exit(1)
    source_a = np.asarray(Image.open(parameters['source_a_file']))[:, :, 0] / 255
    source_b = np.asarray(Image.open(parameters['source_b_file']))[:, :, 0] / 255
    factors = np.stack((source_a, source_b))

    width = parameters['sample_count_width']
    height = parameters['sample_count_height']
    loadings = np.empty((2, height, width))
    one_third = height // 3
    for y in range(one_third):
        loadings[0, y, :] = 1.0
        loadings[1, y, :] = 0.0
    for y in range(one_third, 2*one_third):
        loadings[0, y, :] = 1 - (y - one_third) / one_third
        loadings[1, y, :] = 1 - loadings[0, y, :]
    for y in range(2*one_third, height):
        loadings[0, y, :] = 0.0
        loadings[1, y, :] = 1.0

    return factors, loadings


def save_decomposition(output_dir, method_name, slice_x, slice_y, factors, loadings):
    output_prefix = os.path.join(
            output_dir,
            '{}_{}-{}_{}-{}'.format(
                method_name,
                slice_x.start, slice_x.stop,
                slice_y.start, slice_y.stop))
    # TODO: Do I want to save these as floats?
    factors_scaling = 255.0 / (np.max(factors) or 1)
    loadings_scaling = 255.0 / (np.max(loadings) or 1)
    for i, factor in enumerate(factors):
        Image.fromarray((factor * factors_scaling).astype('uint8')).save('{}_factors_{}.tiff'.format(output_prefix, i))
    for i, loading in enumerate(loadings):
        Image.fromarray((loading * loadings_scaling).astype('uint8')).save('{}_loadings_{}.tiff'.format(output_prefix, i))


def save_object(output_dir, method_name, slice_x, slice_y, data):
    output_filename = os.path.join(
            output_dir,
            '{}_{}-{}_{}-{}.pickle'.format(
                method_name,
                slice_x.start, slice_x.stop,
                slice_y.start, slice_y.stop))
    with open(output_filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_combined_loadings(output_dir):
    loading_image_infos = result_image_file_info(output_dir, 'loadings')

    if len(loading_image_infos) == 0:
        return

    first_image_infos = next(iter(loading_image_infos.values()))
    width  = max([image_info['x_stop'] for image_info in first_image_infos])
    height = max([image_info['y_stop'] for image_info in first_image_infos])

    for method_name, image_infos in loading_image_infos.items():
        merged_loadings = np.zeros((height, width, 3), 'float32')
        for image_info in image_infos:
            factor_index = image_info['factor_index']
            if factor_index >= 3:
                factor_index = 0
                print('WARNING: Too many factors for RGB output. Writing loading to red channel.')
            merged_loadings[image_info['y_start']:image_info['y_stop'],
                            image_info['x_start']:image_info['x_stop'],
                            factor_index] += np.asarray(Image.open(image_info['filename']))

        # TODO: Do I want to save these as floats?
        image_data = (merged_loadings * (255 / np.max(merged_loadings))).astype('uint8')
        Image.fromarray(image_data).save(os.path.join(output_dir, 'loading_map_{}.tiff'.format(method_name)))


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


def data_source_linear_ramp(parameters):
    ground_truth_factors, ground_truth_loadings = generate_test_linear_noiseless(parameters)

    # TODO(simonhog): numpy probably has a way of doing this without the reshape
    factor_count, pattern_height, pattern_width = ground_truth_factors.shape
    loadings_count, sample_height, sample_width = ground_truth_loadings.shape
    factors = ground_truth_factors.reshape((factor_count, -1))
    loadings = ground_truth_loadings.reshape((factor_count, -1))

    split_width = parameters['split_width'] if 'split_width' in parameters else sample_width
    split_height = parameters['split_height'] if 'split_height' in parameters else sample_height

    # TODO(simonhog): Might want to align the splits to data chunk sizes (diffraction_patterns.chunks)
    for split_start_y in range(0, sample_height, split_height):
        split_end_y = min(split_start_y + split_height, sample_height)
        slice_y = slice(split_start_y, split_end_y)
        for split_start_x in range(0, sample_width, split_width):
            split_end_x = min(split_start_x + split_width, sample_width)
            slice_x = slice(split_start_x, split_end_x)
            save_decomposition(parameters['output_dir_run'], 'ground_truth', slice_x, slice_y, ground_truth_factors, ground_truth_loadings[:, slice_y, slice_x], range(factor_count))

    diffraction_patterns = np.matmul(loadings.T, factors)
    diffraction_patterns = diffraction_patterns.reshape((sample_height, sample_width, pattern_height, pattern_width))
    return diffraction_patterns


def data_source_sample_data(parameters):
    sample_filename = parameters['sample_file']
    sample = pxm.load(sample_filename, lazy=True)
    parameters['nav_scale_x'] = sample.axes_manager.navigation_axes[1].scale
    parameters['nav_scale_y'] = sample.axes_manager.navigation_axes[0].scale
    # TODO(simonhog): Parameterize data type?
    sample.change_dtype('float32')
    sample.data *= 1/sample.data.max()
    return sample.data


def data_source_name(data_source):
    return 'data_source_' + data_source


def preprocessor_affine_transform(data, parameters):
    # TODO(simonhog): What is the cost of wrapping in ElectronDiffraction?
    signal = pxm.ElectronDiffraction(data)
    scale_x = parameters['scale_x']
    scale_y = parameters['scale_y']
    offset_x = parameters['offset_x']
    offset_y = parameters['offset_y']
    signal.apply_affine_transformation(np.array([
            [scale_x, 0, offset_x],
            [0, scale_y, offset_y],
            [0, 0, 1]
        ]))
    return signal.data


def preprocessor_gaussian_difference(data, parameters):
    # TODO(simonhog): Does this copy the data? Hopefully not
    signal = pxm.ElectronDiffraction(data)
    sig_width = signal.axes_manager.signal_shape[0]
    sig_height = signal.axes_manager.signal_shape[1]

    # signal.center_direct_beam(
            # radius_start=parameters['center_radius_start'],
            # radius_finish=parameters['center_radius_finish'],
            # square_width=parameters['center_square'],
            # show_progressbar=False)

    signal = signal.remove_background(
            'gaussian_difference',
            sigma_min=parameters['gaussian_sigma_min'],
            sigma_max=parameters['gaussian_sigma_max'])
    signal.data /= signal.data.max()

    # TODO(simonhog): Could cache beam mask between calls
    # sig_center = (sig_width - 1) / 2, (sig_height - 1) / 2
    # direct_beam_mask = circular_mask(shape=(sig_width, sig_height),
                                     # radius=parameters['direct_beam_mask_radius'],
                                     # center=sig_center)
    # np.invert(direct_beam_mask, out=direct_beam_mask)
    # signal.data *= direct_beam_mask

    return signal.data


def preprocessor_hdome(data, parameters):
    signal = pxm.ElectronDiffraction(data)
    signal = signal.remove_background('h-dome', h=parameters['hdome_h'])
    signal.data *= 1/signal.data.max()
    return signal.data


def preprocessor_name(preprocessor):
    return 'preprocessor_' + preprocessor


def run_factorizations(parameters):
    output_dir = parameters['output_dir'] if 'output_dir' in parameters else ''

    if not os.path.exists(os.path.join(output_dir, 'tmp')):
        os.makedirs(os.path.join(output_dir, 'tmp'))

    output_dir = os.path.join(output_dir, 'run_{}_{}'.format(parameters['shortname'], parameters['__date_string']))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    parameters['output_dir_run'] = output_dir

    if not 'data_source' in parameters:
        print('No data_source given')
        exit(1)
    elif data_source_name(parameters['data_source']) not in globals():
        print('Unknown data_source {}'.format(parameters['data_source']))
        exit(1)

    methods = [method.strip() for method in parameters['methods'].split(',')]

    data_source_loader = globals()[data_source_name(parameters['data_source'])]
    if 'preprocess' in parameters:
        preprocessors = [globals()[preprocessor_name(name.strip())] for name in parameters['preprocess'].split(',')]

    diffraction_patterns = data_source_loader(parameters)
    # NOTE(simonhog): Assuming row-major storage
    full_width = diffraction_patterns.shape[1]
    full_height = diffraction_patterns.shape[0]
    split_width = parameters['split_width'] if 'split_width' in parameters else full_width
    split_height = parameters['split_height'] if 'split_height' in parameters else full_height

    # TODO(simonhog): Might want to align the splits to data chunk sizes (diffraction_patterns.chunks)
    for split_start_y in range(0, full_height, split_height):
        split_end_y = min(split_start_y + split_height, full_height)
        slice_y = slice(split_start_y, split_end_y)
        for split_start_x in range(0, full_width, split_width):
            split_end_x = min(split_start_x + split_width, full_width)
            slice_x = slice(split_start_x, split_end_x)
            print('\n\n====================================')
            print('Starting work on slice ({}:{}, {}:{}) of ({} {})\n'.format(
                split_start_x, split_end_x,
                split_start_y, split_end_y,
                full_width, full_height))
            current_data = np.array(diffraction_patterns[slice_y, slice_x])
            if 'preprocess' in parameters:
                print('Preprocessing')
                start_time = time.perf_counter()

                for preprocessor in preprocessors:
                    current_data = preprocessor(current_data, parameters)

                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print('    Elapsed: {}'.format(elapsed_time))
                elapsed_key = '__elapsed_time_preprocessing'
                parameters[elapsed_key] = elapsed_time + (parameters[elapsed_key] if elapsed_key in parameters else 0)

            for method_name in methods:
                print('Running factorizer "{}"'.format(method_name))
                start_time = time.perf_counter()

                factorizer = get_factorizer(method_name)

                save_data, save_method = factorizer(current_data.copy(), parameters)
                factor_indices = []

                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print('    Elapsed: {}'.format(elapsed_time))
                elapsed_key = '__elapsed_time_{}'.format(method_name)
                parameters[elapsed_key] = elapsed_time + (parameters[elapsed_key] if elapsed_key in parameters else 0)
                if save_method == 'object':
                    save_object(output_dir, method_name, slice_x, slice_y, save_data)
                elif save_method == 'decomposition':
                    save_decomposition(output_dir, method_name, slice_x, slice_y, *save_data)
                else:
                    print('WARNING: Save method "{}" unknown. Using decomposition'.format(save_method))
                    save_method = 'decomposition'
                    save_decomposition(output_dir, method_name, slice_x, slice_y, *save_data)

                parameters['__save_method_{}'.format(method_name)] = save_method


                if False:
                    for i in range(factors.shape[0]):
                        plt.subplot(2, factors.shape[0], i+1)
                        plt.imshow(factors[i])
                    for i in range(factors.shape[0]):
                        plt.subplot(2, factors.shape[0], factors.shape[0] + i + 1)
                        plt.imshow(loadings[i])
                    plt.show()

    parameters_save(parameters, output_dir)
    save_combined_loadings(output_dir)


if __name__ == '__main__':
    hs.preferences.General.nb_progressbar = False
    if len(sys.argv) > 1:
        parameters = parameters_parse(sys.argv[1])
    else:
        # TODO(simonhog): Help message
        parameters = {}

    run_factorizations(parameters)
