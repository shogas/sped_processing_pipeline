import os
import sys
import time

import matplotlib
matplotlib.use('Qt5Agg')

from hyperspy.api import load as hs_load
import umap
import hdbscan

from parameters import parameters_parse
from utils.performance_log import LogThread
from utils.performance_log import time_log_call


def calculate_cluster(signal, n_neighbours, min_dist, n_components, min_samples, cluster_size):
    embedding = umap.UMAP(
        n_neighbors =n_neighbours,
        min_dist    =min_dist,
        n_components=n_components,
        random_state=42,
    ).fit_transform(signal)

    clusterer = hdbscan.HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=cluster_size
    ).fit(embedding)


def run_umap(parameters):
    output_dir = parameters['output_dir'] if 'output_dir' in parameters else ''
    output_dir = os.path.join(output_dir, 'run_{}_{}'.format(parameters['shortname'], parameters['__date_string']))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    result_filename = os.path.join(output_dir, 'time.txt')
    log_filename = os.path.join(output_dir, 'mem.txt')

    in_file = parameters['sample_file']
    n_neighbour_step = parameters['n_neighbour_step']
    n_neighbour_max = parameters['n_neighbour_max']
    split_size_step = parameters['split_size_step']
    split_size_max = parameters['split_size_max']

    dp = hs_load(in_file, lazy=True)
    signal_height, signal_width = dp.data.shape[2:4]

    log_thread = LogThread(log_filename)
    log_thread.start()

    with open(result_filename, 'w') as result_file:
        for n_neighbours in range(n_neighbour_step, n_neighbour_max + 1, n_neighbour_step):
            print('Projecting with {} neighbours'.format(n_neighbours))
            for split_size in range(split_size_step, split_size_max + 1, split_size_step):
                width = 100
                height = split_size // width
                print('    Splitting to {}x{}={}'.format(width, height, width*height))
                dp_split = dp.inav[:width, :height]
                dp_split.change_dtype('float')
                dp_split = dp_split.data.compute()
                data_flat = dp_split.reshape(-1, signal_width*signal_height)

                time_elapsed = time_log_call(
                        result_file,
                        lambda: calculate_cluster(data_flat, n_neighbours,
                            parameters['umap_min_dist'],
                            parameters['umap_n_components'],
                            parameters['umap_cluster_min_samples'],
                            parameters['umap_cluster_size']),
                        n_neighbours, split_size)

                print('    End, elapsed: {:.2f}'.format(time_elapsed))

    log_thread.stop()


if __name__ == '__main__':
    parameters = parameters_parse(sys.argv[1])
    run_umap(parameters)
