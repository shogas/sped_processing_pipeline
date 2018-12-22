from hyperspy.signals import Signal2D
import numpy as np

from utils.decomposition import decomposition_ard_so


def process(diffraction_patterns, parameters):
    phase_count = parameters['phase_count']
    rep_count = parameters['rep_count']
    max_iterations = parameters['max_iterations']
    orthogonality_constraint = parameters['orthogonality_constraint']

    s = Signal2D(diffraction_patterns)

    decomposition_ard_so(s,
        n_components=phase_count,
        n_reps=rep_count,
        threshold_merge=0.95,
        max_iterations=max_iterations,
        wo=orthogonality_constraint)

    if s.learning_results.output_dimensions > 0:
        factors = s.get_decomposition_factors().data
        loadings = s.get_decomposition_loadings().data
    else:
        factors = np.array([diffraction_patterns[0, 0]])
        loadings = np.ones((1, diffraction_patterns.data.shape[0], diffraction_patterns.data.shape[1]))

    scale = loadings.max()
    factors *= scale
    loadings /= scale
    return (factors, loadings), 'decomposition'

