from utils.template_matching import generate_diffraction_library
from pyxem import ElectronDiffraction
from pyxem.generators.indexation_generator import IndexationGenerator


def factorize(diffraction_patterns, parameters):
    dps = ElectronDiffraction(diffraction_patterns)

    phase_names = parameters['phase_names']
    diffraction_library = generate_diffraction_library(parameters, phase_names, 'complete')
    pattern_indexer = IndexationGenerator(dps, diffraction_library)
    indexation_results = pattern_indexer.correlate(n_largest=4, keys=phase_names, show_progressbar=False)
    crystallographic_map = indexation_results.get_crystallographic_map(show_progressbar=False)

    return crystallographic_map.data, 'object'

