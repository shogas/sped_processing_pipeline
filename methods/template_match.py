import os

import numpy as np
from diffpy.structure import loadStructure

from pyxem import ElectronDiffraction
from pyxem.generators.diffraction_generator import DiffractionGenerator
from pyxem.generators.library_generator import DiffractionLibraryGenerator
from pyxem.generators.indexation_generator import IndexationGenerator
from pyxem.generators.structure_library_generator import StructureLibraryGenerator
from pyxem.libraries.diffraction_library import load_DiffractionLibrary


def process(diffraction_patterns, parameters):
    """Run template mathing on the diffraction patterns to extract
    phase and orientation data.

    Parameters
    ----------
    diffraction_patterns : numpy.ndarray
        4D numpy array containing the diffraction patterns.
    parameters : dict
        Dictionary of parameters:
            'phase_names' : string
                Comma-separated list of phase names.
            'output_dir' : string
                Output directory, used to store a cache of the diffraction library.
            'shortname' : string
                Identifier for this run.
            'specimen thickness' : float
                Simulation parameter, used to calculate max excitation error.
            'beam_energy_keV' : float
                Simulation parameter, beam energy in keV.
            'reciprocal_angstrom_per_pixel' : float
                Diffraction pattern calibration.
            'rotation_list_resolution' : float
                Angular resolution in degrees. Defaults to 1.
            'phase_<phase name>_structure_file' : string
                Path to cif file for phase.
            'phase_<phase name>_crystal_system' : string
                Crystal system for phase.
            'phase_<phase name>_inplane_rotations' : string
                Comma separated list of inplane rotations in degrees.

            Keys with <phase name> in expect one such key for each phase name
            specified in 'phase_names'.

    Returns
    -------
    results : pyxem.IndexationResults
        Indexation results as returned by pyxem.
    result_type : string
        IndexationResults, to indicate how the results should be treated.
    """
    # Parse the list of phase names and load input data into a pyxem.ElectronDiffraction
    phase_names = [phase_name.strip() for phase_name in parameters['phase_names'].split(',')]
    dp = ElectronDiffraction(diffraction_patterns)

    # Try to load the diffraction library from disk
    diffraction_library_cache_filename = os.path.join(
            parameters['output_dir'],
            'tmp/diffraction_library_{}.pickle'.format(parameters['shortname']))
    if os.path.exists(diffraction_library_cache_filename):
        diffraction_library = load_DiffractionLibrary(diffraction_library_cache_filename, safety=True)
    else:
        # If it does not exist, create one and save it to disk
        diffraction_library = create_diffraction_library(parameters, dp.axes_manager.signal_shape[0])
        diffraction_library.pickle_library(diffraction_library_cache_filename)

    # Set up the indexer and get the indexation results
    pattern_indexer = IndexationGenerator(dp, diffraction_library)
    indexation_results = pattern_indexer.correlate(
            n_largest=4,
            keys=phase_names,
            parallel=False)  # This is slower in parallel

    return indexation_results.data, 'IndexationResults'


def create_diffraction_library(parameters, pattern_size):
    """Create a diffraction library.

    Parameters
    ----------
    parameters : dict
        Dictionary of parameters. See the process function above for details.
    half_pattern_size : int
        Side length in pixels of the generated diffraction patterns
    """
    # Read input parameters
    specimen_thickness = parameters['specimen_thickness']
    beam_energy_keV = parameters['beam_energy_keV']
    reciprocal_angstrom_per_pixel = parameters['reciprocal_angstrom_per_pixel']
    phase_names = [phase_name.strip() for phase_name in parameters['phase_names'].split(',')]
    half_pattern_size = pattern_size // 2
    max_excitation_error = 1/specimen_thickness

    if 'rotation_list_resolution' in parameters:
        rotation_list_resolution = np.deg2rad(parameters['rotation_list_resolution'])
    else:
        rotation_list_resolution = np.deg2rad(1)

    # Create the phase descriptions from the parameters
    phase_descriptions = []
    inplane_rotations = []
    for phase_name in phase_names:
        structure = loadStructure(parameters['phase_{}_structure_file'.format(phase_name)])
        crystal_system = parameters['phase_{}_crystal_system'.format(phase_name)]
        rotations = [float(r.strip()) for r in str(parameters['phase_{}_inplane_rotations'.format(phase_name)]).split(',')]
        phase_descriptions.append((phase_name, structure, crystal_system))
        inplane_rotations.append([np.deg2rad(r) for r in rotations])

    # Create a pyxem.StructureLibrary from the phase descriptions using a
    # stereographic projection.
    structure_library_generator = StructureLibraryGenerator(phase_descriptions)
    structure_library = structure_library_generator.get_orientations_from_stereographic_triangle(
            inplane_rotations, rotation_list_resolution)

    # Set up the diffraction generator from the given parameters
    gen = DiffractionGenerator(beam_energy_keV, max_excitation_error=max_excitation_error)
    library_generator = DiffractionLibraryGenerator(gen)
    reciprocal_radius = reciprocal_angstrom_per_pixel*(half_pattern_size - 1)

    # Finally, actually create the DiffractionLibrary. The library is created
    # without the direct beam since it does not contribute to matching.
    diffraction_library = library_generator.get_diffraction_library(
        structure_library,
        calibration=reciprocal_angstrom_per_pixel,
        reciprocal_radius=reciprocal_radius,
        half_shape=(half_pattern_size, half_pattern_size),
        with_direct_beam=False)

    return diffraction_library
