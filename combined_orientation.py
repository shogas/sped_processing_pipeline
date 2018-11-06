import os
import sys

import numpy as np
import pickle
from tqdm import tqdm as report_progress

from pyxem import CrystallographicMap

from common import result_object_file_info
from parameters import parameters_parse


def combine_crystal_map(object_infos):
    total_width  = max((info['x_stop'] for info in object_infos))
    total_height = max((info['y_stop'] for info in object_infos))

    combined_crystal_map_data = np.zeros((total_width, total_height, 7))

    for object_info in report_progress(object_infos):
        report_progress.write('Tile {}:{}  {}:{} (of {} {})'.format(
            object_info['x_start'], object_info['x_stop'],
            object_info['y_start'], object_info['y_stop'],
            total_width, total_height))

        with open(object_info['filename'], 'rb') as f:
            crystal_map_data = pickle.load(f)

        combined_crystal_map_data[
                object_info['y_start']:object_info['y_stop'],
                object_info['x_start']:object_info['x_stop']] = crystal_map_data

    return CrystallographicMap(combined_crystal_map_data)


def combine_orientations(result_directory):
    parameters = parameters_parse(os.path.join(result_directory, 'metadata.txt'))
    methods = [
            method.strip() for method in parameters['methods'].split(',')
            if parameters['__save_method_{}'.format(method.strip())] == 'object']

    object_infos = result_object_file_info(result_directory)
    if 'template_match' in object_infos:
        crystal_map = combine_crystal_map(object_infos['template_match'])
        print(crystal_map.axes_manager)
        print(crystal_map.data)


if __name__ == '__main__':
    result_directory = sys.argv[1]
    combine_orientations(result_directory)
