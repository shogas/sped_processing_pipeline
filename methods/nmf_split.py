import math

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as matplotimg

from .nmf import decompose_nmf

from pyxem import ElectronDiffraction
from pyxem import DiffractionGenerator
from pyxem import DiffractionLibraryGenerator
from pyxem.generators.indexation_generator import IndexationGenerator
from pyxem.libraries.structure_library import StructureLibrary

import diffpy.structure

from transforms3d.axangles import axangle2mat
from transforms3d.euler import euler2mat
from transforms3d.euler import mat2euler



phase_index_to_name = {
    0: 'GaAs ZB1',
    1: 'GaAs WZ',
}
def classify(diffraction_library, image):
    # TODO(simonhog): Seems like IndexationGenerator can do multiple images at once? (-> better use of the crystal map)
    diffraction_pattern = ElectronDiffraction([[image]])
    # TODO(simonhog): This has support for arbitrary masks
    indexer = IndexationGenerator(diffraction_pattern, diffraction_library)
    # TODO(simonhog): Get from parameters (see below)
    match_results = indexer.correlate(n_largest=2, keys=phase_index_to_name.values())
    crystal_map = match_results.get_crystallographic_map()
    phase = int(crystal_map.get_phase_map().data[0,0])
    angles = crystal_map.get_modal_angles()[0]
    return phase, angles


def generate_rotation_list(h, k, l, max_theta, resolution):
    # NOTE(simonhog): This is copy-pasted from the sped_nn_recognition codebase
    #                 Should be moved to a common library, the projects should
    #                 be combined, or this utility integrated in pyxem.
    # NOTE(simonhog): Don't edit this function here, instead do it in the other codebase...
    # Assuming cubic
    angle = math.acos((h*0 + k*0 + l*1) / math.sqrt((h**2 + k**2 + l**2)*(0**2 + 0**2 + 1**2)))
    axis = np.cross(np.array([0, 0, 1]), np.array([h, k, l]))
    if np.count_nonzero(axis) == 0:
        axis = np.array([0, 0, 1])

    lattice_to_zone = np.identity(4)
    lattice_to_zone[0:3, 0:3] = axangle2mat(axis, angle)
    zone_to_rotation = np.identity(4)

    # This generates rotations around the given axis, with a denser sampling close to the axis
    resolution = np.deg2rad(resolution)
    min_psi = -np.pi
    max_psi = np.pi
    rotations = np.empty((math.ceil(max_theta / resolution), math.ceil((max_psi - min_psi) / resolution), 3))
    for i, theta in enumerate(np.arange(0, max_theta, resolution)):
        theta_rot = euler2mat(0, theta, 0, 'rzxz')
        for j, psi in enumerate(np.arange(min_psi, max_psi, resolution)):
            zone_to_rotation[0:3, 0:3] = np.matmul(theta_rot, euler2mat(0, 0, psi, 'rzxz'))
            lattice_to_rotation = np.matmul(lattice_to_zone, zone_to_rotation)
            rotations[i, j] = np.rad2deg(mat2euler(lattice_to_rotation, 'rzxz'))

    return rotations


# TODO(simonhog): A lot of this comes from the sped_nn_recognition codebase
def generate_diffraction_library(parameters):
    h = int(parameters['zone_h'])
    k = int(parameters['zone_k'])
    l = int(parameters['zone_l'])
    beam_energy_keV = float(parameters['beam_energy_keV'])
    specimen_thickness = float(parameters['specimen_thickness'])
    target_pattern_dimension_pixels = int(parameters['target_pattern_dimension_pixels'])
    reciprocal_radius = float(parameters['reciprocal_radius'])
    angstrom_per_pixel = float(parameters['angstrom_per_pixel'])
    max_theta = np.pi / 8
    resolution = 5
    rotation_list = generate_rotation_list(h, k, l, max_theta, resolution).reshape(-1, 3)
    # TODO(simonhog): Generalize to use (arrays) from parameter file
    # TODO(simonhog): Figure out how diffpy actually want absolute paths on Windows
    structure_zb = diffpy.structure.loadStructure('file:///' + parameters['structure_zb_file'])
    structure_wz = diffpy.structure.loadStructure('file:///' + parameters['structure_wz_file'])
    structure_library = StructureLibrary(
            ['GaAs ZB1', 'GaAs WZ'],
            [structure_zb, structure_wz],
            [rotation_list, rotation_list])

    diffraction_generator = DiffractionGenerator(beam_energy_keV, max_excitation_error = 1/specimen_thickness)
    library_generator = DiffractionLibraryGenerator(diffraction_generator)
    return library_generator.get_diffraction_library(
            structure_library,
            calibration=angstrom_per_pixel,
            reciprocal_radius=reciprocal_radius,
            half_shape=(target_pattern_dimension_pixels / 2, target_pattern_dimension_pixels / 2),
            with_direct_beam=True)  # TODO(simonhog): Probably want to mask this in the actual



def factorize(diffraction_patterns, parameters):
    """ Factorize the diffraction patterns using NMF on slices.

    The diffraction patterns are divided into slices, and each of them are then
    decomposed using NMF. The resulting factors matched against a library of
    patterns and then scaled to also match the maximum intensity. This same
    scaling is used to scale the loading, using and correcting for the fact
    that a factorization X = FL only is unique to a factor s, where F -> F' =
    sF and L -> L' = L/s.

    NOTE: Current implementation of pattern matching is temporary.
    """

    diffraction_library = generate_diffraction_library(parameters)

    diffraction_pattern_image_size = diffraction_patterns.shape[2]
    full_width = diffraction_patterns.shape[1]
    full_height = diffraction_patterns.shape[0]
    split_width = int(parameters['split_width']) if 'split_width' in parameters else full_width
    split_height = int(parameters['split_height']) if 'split_height' in parameters else full_height
    factor_count = 2  # TODO(simonhog): Automatic/parameterized

    # TODO(simonhog): Constants...
    peak_sigma = 0.02
    max_reciprocal = 1

    loadings = np.empty((factor_count, diffraction_patterns.shape[0], diffraction_patterns.shape[1]))
    for split_start_y in range(0, full_height, split_height):
        split_end_y = min(split_start_y + split_height, full_height)
        slice_y = slice(split_start_y, split_end_y)
        for split_start_x in range(0, full_width, split_width):
            split_end_x = min(split_start_x + split_width, full_width)
            slice_x = slice(split_start_x, split_end_x)
            el_diff = ElectronDiffraction(diffraction_patterns[slice_y, slice_x])
            decompose_nmf(el_diff, factor_count)

            # TODO(simonhog): Use template-matching instead
            new_factors = el_diff.get_decomposition_factors().data
            loading_mappings = { new_factor_index: classify(diffraction_library, new_factors[new_factor_index])
                                    for new_factor_index in range(factor_count) }
            loading_data = el_diff.get_decomposition_loadings().data
            for new_factor_index, (factor_index, angles) in loading_mappings.items():
                if factor_index is not None:
                    phase = diffraction_library[phase_index_to_name[factor_index]]
                    # TODO(simonhog): Use actual peak data instead of simulating the image
                    pattern = phase[tuple(angles)]['Sim'].as_signal(diffraction_pattern_image_size, peak_sigma, max_reciprocal)
                    scaling = np.max(new_factors[new_factor_index]) / np.max(pattern.data)
                    loadings[factor_index, slice_y, slice_x] = scaling * loading_data[new_factor_index]

            total_loading = np.sum(loadings[:, slice_y, slice_x], axis=0)
            for factor_index in range(factor_count):
                loadings[factor_index, slice_y, slice_x] /= total_loading

    factors = []
    for phase_name in phase_index_to_name.values():
        phase = diffraction_library[phase_name]
        # TODO(simonhog): Use actual peak data instead of simulating the image
        pattern = phase[next(iter(phase.keys()))]['Sim'].as_signal(diffraction_pattern_image_size, peak_sigma, max_reciprocal)
        factors.append(pattern.data)

    return factors, loadings

