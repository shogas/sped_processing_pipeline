import os

import numpy as np
from diffpy.structure import loadStructure

from pyxem import ElectronDiffraction
from pyxem.generators.diffraction_generator import DiffractionGenerator
from pyxem.generators.library_generator import DiffractionLibraryGenerator
from pyxem.generators.indexation_generator import IndexationGenerator
from pyxem.generators.structure_library_generator import StructureLibraryGenerator
from pyxem.libraries.diffraction_library import load_DiffractionLibrary


def factorize(diffraction_patterns, parameters):
    dps = ElectronDiffraction(diffraction_patterns)

    phase_names = [phase_name.strip() for phase_name in parameters['phase_names'].split(',')]

    diffraction_library_cache_filename = os.path.join(
            parameters['output_dir'],
            'tmp/diffraction_library_{}.pickle'.format(parameters['shortname']))

    if os.path.exists(diffraction_library_cache_filename):
        diffraction_library = load_DiffractionLibrary(diffraction_library_cache_filename, safety=True)
    else:
        diffraction_library = create_diffraction_library(parameters, dps.axes_manager.signal_shape[0] // 2)
        diffraction_library.pickle_library(diffraction_library_cache_filename)

    pattern_indexer = IndexationGenerator(dps, diffraction_library)
    indexation_results = pattern_indexer.correlate(n_largest=4, keys=phase_names, parallel=False)

    return indexation_results.data, 'object'


def create_diffraction_library(parameters, half_pattern_size):
    specimen_thickness = parameters['specimen_thickness']
    beam_energy_keV = parameters['beam_energy_keV']
    reciprocal_angstrom_per_pixel = parameters['reciprocal_angstrom_per_pixel']
    phase_names = [phase_name.strip() for phase_name in parameters['phase_names'].split(',')]
    rotation_list_resolution = np.deg2rad(1)  # TODO(simonhog): Parameterize

    phase_descriptions = []
    inplane_rotations = []
    for phase_name in phase_names:
        structure = loadStructure(parameters['phase_{}_structure_file'.format(phase_name)])
        crystal_system = parameters['phase_{}_crystal_system'.format(phase_name)]
        rotations = [float(r.strip()) for r in str(parameters['phase_{}_inplane_rotations'.format(phase_name)]).split(',')]
        phase_descriptions.append((phase_name, structure, crystal_system))
        inplane_rotations.append([np.deg2rad(r) for r in rotations])


    structure_library_generator = StructureLibraryGenerator(phase_descriptions)
    structure_library = structure_library_generator.get_orientations_from_stereographic_triangle(
            inplane_rotations, rotation_list_resolution)
    max_excitation_error = 1/specimen_thickness
    gen = DiffractionGenerator(beam_energy_keV, max_excitation_error=max_excitation_error)
    library_generator = DiffractionLibraryGenerator(gen)
    reciprocal_radius = reciprocal_angstrom_per_pixel*(half_pattern_size - 1)

    diffraction_library = library_generator.get_diffraction_library(
        structure_library,
        calibration=reciprocal_angstrom_per_pixel,
        reciprocal_radius=reciprocal_radius,
        half_shape=(half_pattern_size, half_pattern_size),
        with_direct_beam=False)

    return diffraction_library
