import os
import sys

import numpy as np
import hyperspy.api as hs

def factorize(diffraction_patterns, parameters):
    # TODO(simonhog): Very similar to methods/nmf.py

    s = hs.signals.Signal2D(diffraction_patterns).as_lazy()

    # TODO(simonhog): Parameterize
    factor_count = 2
    # TODO(simonhog): Why can't I send show_progressbar=False?
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        s.decomposition(
                True,
                algorithm='ONMF',
                output_dimension=factor_count)
        sys.stdout = old_stdout

    factors = s.get_decomposition_factors().data
    loadings = s.get_decomposition_loadings().data

    scale = loadings.max()
    factors *= scale
    loadings /= scale
    # TODO(simonhog): Handle dask arrays further down the pipeline. Probably move everything to use it? Test performance overhead.
    return (np.asarray(factors), np.asarray(loadings)), 'decomposition'
