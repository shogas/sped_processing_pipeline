import numpy as np

import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from pyxem import ElectronDiffraction

def factorize(diffraction_patterns):
    dps = ElectronDiffraction(diffraction_patterns)
    dps.plot()
    plt.show()
    return np.array([0]), np.array([0])
