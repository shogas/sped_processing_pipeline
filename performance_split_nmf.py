import os
import sys
import time

import matplotlib
matplotlib.use('Qt5Agg')

import pyxem as pxm

from parameters import parameters_parse
from utils.decomposition import decompose_nmf
from utils.performance_log import LogThread
from utils.performance_log import time_log_call


def run_split(parameters):
    output_dir = parameters['output_dir'] if 'output_dir' in parameters else ''
    output_dir = os.path.join(output_dir, 'run_{}_{}'.format(parameters['shortname'], parameters['__date_string']))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    result_filename = os.path.join(output_dir, 'time.txt')
    log_filename = os.path.join(output_dir, 'mem.txt')

    in_file = parameters['sample_file']
    component_count_step = parameters['component_count_step']
    component_count_max = parameters['component_count_max']
    split_size_step = parameters['split_size_step']
    split_size_max = parameters['split_size_max']

    dp = pxm.load(in_file, lazy=True)

    log_thread = LogThread(log_filename)
    log_thread.start()

    with open(result_filename, 'w') as result_file:
        for component_count in range(component_count_step, component_count_max + 1, component_count_step):
            print('Factoring to {} components'.format(component_count))
            for split_size in range(split_size_step, split_size_max + 1, split_size_step):
                width = 100
                height = split_size // width
                print('    Splitting to {}x{}={}'.format(width, height, width*height))
                dp_split = pxm.ElectronDiffraction(dp.inav[:width, :height])
                dp_split.change_dtype('float')

                time_elapsed = time_log_call(
                        result_file,
                        lambda: decompose_nmf(dp_split, component_count),
                        component_count, split_size)

                print('    End, elapsed: {:.2f}'.format(time_elapsed))

    log_thread.stop()


if __name__ == '__main__':
    parameters = parameters_parse(sys.argv[1])
    run_split(parameters)
