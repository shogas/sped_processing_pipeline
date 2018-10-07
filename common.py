import glob
import os
import re

def result_image_file_info(dir, type):
    filename_regex = re.compile(r"""(?P<method_name>.*)_
                                    (?P<x_start>\d*)-(?P<x_stop>\d*)_
                                    (?P<y_start>\d*)-(?P<y_stop>\d*)_
                                    {}_(?P<factor_index>\d*)\.tiff""".format(type), re.X)
    loadings_filenames = glob.iglob(os.path.join(dir, '*{}_*.tiff'.format(type)))
    image_infos = {}
    for loading_filename in loadings_filenames:
        match = filename_regex.match(os.path.basename(loading_filename))
        method_name = match.group('method_name')
        if method_name not in image_infos:
            image_infos[method_name] = []
        image_infos[method_name].append(
        {
            'filename':     os.path.join(dir, match.group(0)),
            'x_start':      int(match.group('x_start')),
            'x_stop':       int(match.group('x_stop')),
            'y_start':      int(match.group('y_start')),
            'y_stop':       int(match.group('y_stop')),
            'factor_index': int(match.group('factor_index')),
        })

    return image_infos

