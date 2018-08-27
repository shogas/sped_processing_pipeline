import sys

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.image as matplotimg
import matplotlib.pyplot as plt

from parameters import parseParameters, saveParameters

def linearBlend(source_a, source_b, sample_count):
    """ Blend of two sources, where the first third is entirely source a, the
        last third entirely source b and the middle is a linear blend.

        source_a: Array of values representing one source
        source_b: Array of values representing the other source
        sample_count: Number of samples to return

        Result: Generator yielding the specified values.
    """

    one_third = sample_count // 3
    for i in range(one_third):
        yield source_a
    for i in range(1, one_third + 1):
        t = i / one_third
        yield (1-t)*source_a + t*source_b
    for i in range(sample_count - 2*one_third):
        yield source_b

def noiselessRun(parameters, factorizer):
    source_a = matplotimg.imread(parameters['source_a_file'])
    source_b = matplotimg.imread(parameters['source_b_file'])
    diffraction_patterns = linearBlend(source_a, source_b, int(parameters['sample_count']))
    return factorizer(diffraction_patterns)

def debugFactorizer(diffraction_patterns):
    index = 4
    for i in range(index):
        next(diffraction_patterns)
    asdf = next(diffraction_patterns)
    print(asdf.shape)
    plt.imshow(asdf, cmap='gray')
    plt.show()

def main(parameter_file):
    run_parameters = parseParameters(parameter_file)
    noiselessRun(run_parameters, debugFactorizer)
    saveParameters(run_parameters, '../../Data/Tmp')

if __name__ == '__main__':
    main(sys.argv[1])
