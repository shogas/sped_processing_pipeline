from datetime import datetime
from heapq import nlargest
from operator import itemgetter
import os
import sys
import time
import threading

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import hyperspy.api as hs

import pyxem as pxm
from pyxem import DiffractionGenerator
from pyxem import DiffractionLibraryGenerator
from pyxem.libraries.structure_library import StructureLibrary
from pyxem.utils.sim_utils import rotation_list_stereographic
import diffpy.structure

import winstats


from parameters import parameters_parse


def correlate(image, pattern_dictionary):
    pattern_intensities = pattern_dictionary['intensities']
    pixel_coordinates = pattern_dictionary['pixel_coords']
    pattern_normalization = pattern_dictionary['pattern_norm']

    # The x,y choice here is correct. Basically the numpy/hyperspy conversion is a danger
    image_intensities = image[pixel_coordinates[:, 1], pixel_coordinates[:, 0]]

    return np.dot(image_intensities, pattern_intensities) / pattern_normalization


# From before pyxem ef28af2077f8c456cfcc1f92b768878fcf539f58
def correlate_library_old(image, library, n_largest, mask, keys=[]):
    i = 0
    out_arr = np.zeros((n_largest * len(library), 5))
    if mask == 1:
        for key in library.keys():
            correlations = dict()
            for orientation, diffraction_pattern in library[key].items():
                correlation = correlate(image, diffraction_pattern)
                correlations[orientation] = correlation
                res = nlargest(n_largest, correlations.items(),
                               key=itemgetter(1))
            for j in np.arange(n_largest):
                out_arr[j + i * n_largest][0] = i
                out_arr[j + i * n_largest][1] = res[j][0][0]
                out_arr[j + i * n_largest][2] = res[j][0][1]
                out_arr[j + i * n_largest][3] = res[j][0][2]
                out_arr[j + i * n_largest][4] = res[j][1]
            i = i + 1

    else:
        for j in np.arange(n_largest):
            for k in [0, 1, 2, 3, 4]:
                out_arr[j + i * n_largest][k] = np.nan
        i = i + 1
    return out_arr


def correlate_library_new(image, library, n_largest, mask, keys=[]):
    out_arr = np.zeros((len(library), n_largest, 5))
    if mask == 1:
        for phase_index, key in enumerate(library.keys()):
            correlations = np.empty((len(library[key]), 4))
            correlations[:, 0:3] = np.asarray([*library[key].keys()])
            for i, (orientation, diffraction_pattern) in enumerate(library[key].items()):
                correlations[i, 3] = correlate(image, diffraction_pattern)
            res = correlations[correlations[:, 3].argpartition(-n_largest)[-n_largest:]]
            res = res[res[:, 3].argsort()][::-1]
            out_arr[phase_index, :, 0] = phase_index
            out_arr[phase_index, :, 1:] = res
    else:
        out_arr.fill(np.nan)
    return out_arr.reshape((len(library) * n_largest, 5))


def current_timestamp():
    return '{0:%Y}{0:%m}{0:%d}_{0:%H}_{0:%M}_{0:%S}_{0:%f}'.format(datetime.now())


class LogThread(threading.Thread):
    def __init__(self, log_filename, stopper):
        super().__init__()
        self.log_filename = log_filename
        self.stopper = stopper
        self.instance_number = python_process_instance_number(os.getpid())


    def run(self):
        with open(self.log_filename, 'w') as log_file:
            while not self.stopper.is_set():
                private_bytes = winstats.get_perf_data(
                    r'\Process(python{})\Private Bytes'.format(self.instance_number),
                   fmts='long')[0]
                log_file.write('{}\t{}\n'.format(current_timestamp(), private_bytes))
                time.sleep(0.1)


def run_correlation(parameters):
    output_dir = parameters['output_dir'] if 'output_dir' in parameters else ''
    output_dir = os.path.join(output_dir, 'run_{}_{}'.format(parameters['shortname'], parameters['__date_string']))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    result_filename = os.path.join(output_dir, 'time.txt')
    log_filename = os.path.join(output_dir, 'mem.txt')

    structure_filename = parameters['structure_filename']
    in_file = parameters['sample_file']
    beam_energy_keV = parameters['beam_energy_keV']
    specimen_thickness = parameters['specimen_thickness']
    reciprocal_angstrom_per_pixel = parameters['reciprocal_angstrom_per_pixel']
    library_size_max = parameters['library_size_max']
    library_size_step = parameters['library_size_step']

    dp = pxm.load(in_file)
    dp = pxm.ElectronDiffraction(dp.inav[:, dp.data.shape[1] // 2])

    structure = diffpy.structure.loadStructure(structure_filename)

    target_pattern_dimension_pixels = dp.axes_manager.signal_shape[0]
    half_pattern_size = target_pattern_dimension_pixels // 2
    reciprocal_radius = reciprocal_angstrom_per_pixel*(half_pattern_size - 1)
    phase_names = ['ZB']
    # Generate "enough" rotations
    rotation_list = rotation_list_stereographic(
            structure,
            (0, 0, 1), (1, 0, 1), (1, 1, 1),
            [0, np.pi/8, np.pi/7, np.pi/6, np.pi/5],
            np.deg2rad(0.5))
    if len(rotation_list) <= library_size_max:
        print('ERROR: To few rotations in rotation list for max library size {}'.format(library_size_max))
        exit(1)

    nav_shape = dp.axes_manager.navigation_shape
    mask = hs.signals.Signal1D(np.ones((nav_shape[0], nav_shape[1], 1)))

    correlation_methods = [
        ('old', correlate_library_old),
        ('new', correlate_library_new)]

    log_stopper = threading.Event()
    log_thread = LogThread(log_filename, log_stopper).start()

    with open(result_filename, 'w') as result_file:
        for library_size in range(library_size_step, library_size_max + 1, library_size_step):
            structure_library = StructureLibrary(
                    phase_names,
                    [structure],
                    [rotation_list[:library_size]])

            diffraction_generator = DiffractionGenerator(beam_energy_keV, max_excitation_error = 1/specimen_thickness)
            library_generator = DiffractionLibraryGenerator(diffraction_generator)
            print('Generating library with {} entries'.format(library_size))

            diffraction_library = library_generator.get_diffraction_library(
                    structure_library,
                    calibration=reciprocal_angstrom_per_pixel,
                    reciprocal_radius=reciprocal_radius,
                    half_shape=(half_pattern_size, half_pattern_size),
                    with_direct_beam=False)


            for method_name, correlate_library in correlation_methods:
                print('    Correlating with {}'.format(method_name))
                time_start = time.process_time()
                matches = dp.map(correlate_library,
                                     library=diffraction_library,
                                     n_largest=5,
                                     keys=[],
                                     mask=mask,
                                     inplace=False,
                                     parallel=False,
                                     show_progressbar=False)
                time_end = time.process_time()
                time_elapsed = time_end - time_start
                print('    End, elapsed: {:.2f}'.format(time_elapsed))
                result_file.write('{}\t{}\t{}\t{}\n'.format(current_timestamp(), method_name, library_size, time_elapsed))

    log_stopper.set()


def python_process_instance_number(pid):
    instance_number = 0
    while True:
        try:
            python_pid = winstats.get_perf_data(r'\Process(python{})\ID Process'.format(
                '' if instance_number == 0 else '#{}'.format(instance_number)),
                fmts='long')[0]
            if pid == python_pid:
                return '' if instance_number == 0 else '#{}'.format(instance_number)
            instance_number += 1
        except OSError:
            print('Did not find PID after {} attempts'.format(instance_number))
            exit(0)


if __name__ == '__main__':
    parameters = parameters_parse(sys.argv[1])
    run_correlation(parameters)
