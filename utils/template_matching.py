import math
import os

import numpy as np

from pyxem import ElectronDiffraction
from pyxem import DiffractionGenerator
from pyxem import DiffractionLibraryGenerator
from pyxem.generators.indexation_generator import IndexationGenerator
from pyxem.libraries.structure_library import StructureLibrary
from pyxem.libraries.diffraction_library import load_DiffractionLibrary

import diffpy.structure

from transforms3d.axangles import axangle2mat
from transforms3d.euler import euler2axangle
from transforms3d.euler import euler2mat
from transforms3d.euler import mat2euler




def classify(diffraction_library, image, phase_index_to_name):
    # TODO(simonhog): Seems like IndexationGenerator can do multiple images at once? (-> better use of the crystal map)
    diffraction_pattern = ElectronDiffraction([[image]])
    # TODO(simonhog): This has support for arbitrary masks
    indexer = IndexationGenerator(diffraction_pattern, diffraction_library)
    # TODO(simonhog): Get from parameters (see below)
    match_results = indexer.correlate(n_largest=2, keys=phase_index_to_name.values(), show_progressbar=False)
    crystal_map = match_results.get_crystallographic_map(show_progressbar=False)
    phase = int(crystal_map.get_phase_map().data[0,0])
    angles = crystal_map.get_modal_angles()[0]
    return phase, angles


def angle_between_directions(structure,
                            direction1,
                            direction2):
    """Returns the angle in radians between two crystal directions in the given structure."""

    a = structure.lattice.a
    b = structure.lattice.b
    c = structure.lattice.c
    alpha = np.deg2rad(structure.lattice.alpha)
    beta = np.deg2rad(structure.lattice.beta)
    gamma = np.deg2rad(structure.lattice.gamma)

    u1 = direction1[0]
    v1 = direction1[1]
    w1 = direction1[2]

    u2 = direction2[0]
    v2 = direction2[1]
    w2 = direction2[2]

    L = a**2*u1*u2 + b**2*v1*v2 + c**2*w1*w2 \
        + b*c*(v1*w2 + w1*v2)*math.cos(alpha) \
        + a*c*(w1*u2 + u1*w2)*math.cos(beta) \
        + a*b*(u1*v2 + v1*u2)*math.cos(gamma)

    I1 = np.sqrt(a**2 * u1**2 + b**2*v1**2 + c**2*w1**2 \
        + 2*b*c*v1*w1*math.cos(alpha) \
        + 2*a*c*w1*u1*math.cos(beta) \
        + 2*a*b*u1*v1*math.cos(gamma))

    I2 = np.sqrt(a**2 * u2**2 + b**2*v2**2 + c**2*w2**2 \
        + 2*b*c*v2*w2*math.cos(alpha) \
        + 2*a*c*w2*u2*math.cos(beta) \
        + 2*a*b*u2*v2*math.cos(gamma))

    return math.acos(L/(I1*I2))

def generate_rotation_list_euler(structure, h, k, l, max_theta, resolution):
    rotation_list = generate_rotation_list(structure, h, k, l, max_theta, resolution)
    return np.rad2deg([mat2euler(rotation_matrix, 'rzxz') for rotation_matrix in rotation_list])


def generate_rotation_list(structure, h, k, l, max_theta, resolution):
    # NOTE(simonhog): This is copy-pasted from the sped_nn_recognition codebase
    #                 Should be moved to a common library, the projects should
    #                 be combined, or this utility integrated in pyxem.
    # NOTE(simonhog): Don't edit this function here, instead do it in the other codebase...
    # Assuming cubic
    # TODO(simonhog): Symmetry considerations

    zone_to_rotation = np.identity(3)
    lattice_to_zone = np.identity(3)

    angle = angle_between_directions(structure, (0, 0, 1), (h, k, l))
    axis = np.cross(np.array([0, 0, 1]), np.array([h, k, l]))
    if np.count_nonzero(axis) == 0:
        axis = np.array([0, 0, 1])

    lattice_to_zone = axangle2mat(axis, angle)

    # This generates rotations around the given axis, with a denser sampling close to the axis
    min_psi = -np.pi  # TODO(simonhog): Can probably be 0, symmetry always at least 2?
    max_psi = np.pi
    min_phi = 0
    max_phi = np.pi
    theta_count = math.ceil(max_theta / resolution)
    psi_count = math.ceil((max_psi - min_psi) / resolution)
    phi_count = math.ceil((max_phi - min_phi) / resolution)
    rotations = np.empty((theta_count, psi_count, phi_count, 3, 3))
    print(rotations.shape)
    exit(0)
    for i, local_theta in enumerate(np.linspace(0, max_theta, theta_count)):
        for j, local_psi in enumerate(np.linspace(min_psi, max_psi, psi_count)):
            for k, local_phi in enumerate(np.linspace(min_phi, max_phi, phi_count)):
                # TODO: Check that this is the correct order. Swap phi/psi?
                zone_to_rotation = euler2mat(local_phi, local_theta, local_psi, 'sxyz')
                lattice_to_rotation = np.matmul(lattice_to_zone, zone_to_rotation)
                rotations[i, j, k] = lattice_to_rotation

    return rotations.reshape(-1, 3, 3)


def generate_fibonacci_spiral_euler(structure, h, k, l, max_theta, resolution):
    rotation_list = generate_fibonacci_spiral(structure, h, k, l, max_theta, resolution)
    return np.rad2deg([mat2euler(rotation_matrix, 'rzxz') for rotation_matrix in rotation_list])


def generate_fibonacci_spiral(structure, h, k, l, max_theta, resolution):
    # Vogel's method -> disk. Modify -> sphere surface
    zone_to_rotation = np.identity(3)
    lattice_to_zone = np.identity(3)

    angle = angle_between_directions(structure, (0, 0, 1), (h, k, l))
    axis = np.cross(np.array([0, 0, 1]), np.array([h, k, l]))
    if np.count_nonzero(axis) == 0:
        axis = np.array([0, 0, 1])

    lattice_to_zone = axangle2mat(axis, angle)

    n = 50

    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1, np.cos(max_theta), n)

    radius = np.sqrt(1 - z**2)

    points = np.zeros((n, 3))
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = z

    phi_min = 0
    phi_max = np.pi
    phi_count = math.ceil((phi_max - phi_min) / resolution)
    rotations = np.empty((n, phi_count, 3, 3))
    for i, point in enumerate(points):
        # Simplifications to cos angle formula since one of the directions is (0, 0, 1)
        point_angle = math.acos(point[2]/np.linalg.norm(point))
        point_axis = np.cross(np.array([0, 0, 1]), np.array([h, k, l]))
        if np.count_nonzero(point_axis) == 0:
            point_axis = np.array([0, 0, 1])
        point_rotation = axangle2mat(point_axis, point_angle)
        for k, local_phi in enumerate(np.linspace(phi_min, phi_max, phi_count)):
            phi_rotation = euler2mat(local_phi, 0, 0, 'rzxz')
            zone_to_rotation = np.matmul(point_rotation, phi_rotation)
            rotations[i, k] = np.matmul(lattice_to_zone, zone_to_rotation)

    return rotations.reshape(-1, 3, 3)

    if False:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(1, figsize=plt.figaspect(1)*2)
        ax = fig.gca(projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=theta, cmap='viridis')
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        plt.show()


def generate_complete_rotation_list_euler(structure, corner_a, corner_b, corner_c, resolution, psi_angles):
    rotation_list = generate_complete_rotation_list(structure, corner_a, corner_b, corner_c, resolution, psi_angles)
    return np.rad2deg([mat2euler(rotation_matrix, 'rzxz') for rotation_matrix in rotation_list])


# NOTE(simonhog): This code is duplicated across multiple projects now
def generate_complete_rotation_list(structure, corner_a, corner_b, corner_c, resolution, psi_angles):
    """Generate a rotation list covering the inverse pole figure specified by three
        corners in cartesian coordinates.

    Arguments:
        structure: diffpy.structure.Structure, used for calculating angles
        corner_a, corner_b, corner_c: The three corners of the inverse pole
            figure given by three coordinates in the coordinate system
            specified by the structure lattice.
        resolution: Angular resolution in radians of the generated rotation
            list.
        psi_angles: np.array with angles in radians for final rotation angle.

    Returns:
        Rotations covering the inverse pole figure given as a `np.array` of Euler
            angles in degress. This `np.array` can be passed directly to pyxem.
    """

    # Start defining some angles and normals from the given corners
    angle_a_to_b = angle_between_directions(structure, corner_a, corner_b)
    angle_a_to_c = angle_between_directions(structure, corner_a, corner_c)
    angle_b_to_c = angle_between_directions(structure, corner_b, corner_c)
    axis_a_to_b = np.cross(corner_a, corner_b)
    axis_a_to_c = np.cross(corner_a, corner_c)

    # Input validation. The corners have to define a non-degenerate triangle
    if np.count_nonzero(axis_a_to_b) == 0:
        raise ValueError("Directions a and b are parallel")
    if np.count_nonzero(axis_a_to_c) == 0:
        raise ValueError("Directions a and c are parallel")


    # Find the maxiumum number of points we can generate, given by the
    # resolution, then allocate storage for them. For the theta direction,
    # ensure that we keep the resolution also along the direction to the corner
    # b or c farthest away from a.
    psi_count = len(psi_angles)
    theta_count = math.ceil(max(angle_a_to_b, angle_a_to_c) / resolution)
    phi_count = math.ceil(angle_b_to_c / resolution)
    rotations = np.zeros((theta_count, phi_count, psi_count, 3, 3))

    # For each theta_count angle theta, evenly spaced
    for i, (theta_b, theta_c) in enumerate(
            zip(np.linspace(0, angle_a_to_b, theta_count),
                np.linspace(0, angle_a_to_c, theta_count))):
        # Define the corner local_b at a rotation theta from corner_a toward
        # corner_b on the circle surface. Similarly, define the corner local_c
        # at a rotation theta from corner_a toward corner_c

        rotation_a_to_b = axangle2mat(axis_a_to_b, theta_b)
        rotation_a_to_c = axangle2mat(axis_a_to_c, theta_c)
        local_b = np.dot(corner_a, rotation_a_to_b)
        local_c = np.dot(corner_a, rotation_a_to_c)

        # Then define an axis and a maximum rotation to create a great cicle
        # arc between local_b and local_c. Ensure that this is not a degenerate
        # case where local_b and local_c are coincident.
        angle_local_b_to_c = angle_between_directions(structure, local_b, local_c)
        axis_local_b_to_c = np.cross(local_b, local_c)
        if np.count_nonzero(axis_local_b_to_c) == 0:
            # Theta rotation ended at the same position. First position, might
            # be other cases?
            axis_local_b_to_c = corner_a
        axis_local_b_to_c /= np.linalg.norm(axis_local_b_to_c)


        # Generate points along the great circle arc with a distance defined by
        # resolution.
        phi_count_local = math.ceil(angle_local_b_to_c / resolution)
        for j, phi in enumerate(np.linspace(0, angle_local_b_to_c, phi_count_local)):
            rotation_phi = axangle2mat(axis_local_b_to_c, phi)
            for k, psi in enumerate(psi_angles):
                rotation_psi = axangle2mat((0, 0, 1), psi)

                # Combine the rotations. Order is important. The structure is
                # multiplied from the left in diffpy, and we want to rotate by
                # phi first.
                rotation = np.matmul(rotation_phi, np.matmul(rotation_a_to_b, rotation_psi))
                # Finally, convert to Euler angles in degrees for passing
                # to pyxem (which then immediately converts them back to
                # a matrix).
                rotations[i, j, k] = rotation

    # We also remove duplicates before returning. This eliminates the unused rotations.
    return np.unique(rotations.reshape(-1, 3, 3), axis=0)


def uvtw_to_uvw(u, v, t, w):
    U, V, W = 2*u + v, 2*v + u, w
    common_factor = math.gcd(math.gcd(U, V), W)
    return tuple((int(x/common_factor)) for x in (U, V, W))


def direction_to_cartesian(structure_from, direction_from):
    # From formula for change of basis, see hand written description
    a = structure_from.lattice.a
    b = structure_from.lattice.b
    c = structure_from.lattice.c
    alpha = np.deg2rad(structure_from.lattice.alpha)  # angle a to c
    beta  = np.deg2rad(structure_from.lattice.beta)    # angle b to c
    gamma = np.deg2rad(structure_from.lattice.gamma)  # angle a to b

    cos_alpha = math.cos(alpha)
    cos_beta = math.cos(beta)
    cos_gamma = math.cos(gamma)
    sin_gamma = math.sin(gamma)

    transform_e1 = np.array([
        a,
        0,
        0])
    transform_e2 = np.array([
        b*cos_gamma,
        b*sin_gamma,
        0])

    factor_e3_0 = cos_beta
    factor_e3_1 = (cos_alpha - cos_beta*cos_gamma)/sin_gamma
    assert(np.dot(factor_e3_0, factor_e3_0) + np.dot(factor_e3_1, factor_e3_1) < 1)  # TODO: Temporary?
    factor_e3_2 = math.sqrt(1 - np.dot(factor_e3_0, factor_e3_0) - np.dot(factor_e3_1, factor_e3_1))
    transform_e3 = np.array([
        c*factor_e3_0,
        c*factor_e3_1,
        c*factor_e3_2])

    transform = np.array([
        transform_e1,
        transform_e2,
        transform_e3]).T

    return np.dot(transform, direction_from)


def plot_debug(parameters):
    beam_energy_keV = parameters['beam_energy_keV']
    specimen_thickness = parameters['specimen_thickness']
    target_pattern_dimension_pixels = parameters['target_pattern_dimension_pixels']
    reciprocal_angstrom_per_pixel = parameters['reciprocal_angstrom_per_pixel']
    # TODO(simonhog): Parameterize
    rotation_list_resolution = np.deg2rad(1)
    # TODO(simonhog): Generalize to use (arrays) from parameter file
    # TODO(simonhog): Figure out how diffpy actually want absolute paths on Windows
    structure_zb = diffpy.structure.loadStructure('file:///' + parameters['structure_zb_file'])
    structure_wz = diffpy.structure.loadStructure('file:///' + parameters['structure_wz_file'])

    import matplotlib.pyplot as plt
    plt.figure(51)
    plt.cla()
    diffraction_generator = DiffractionGenerator(beam_energy_keV, max_excitation_error = 1/specimen_thickness)
    half_pattern_size = target_pattern_dimension_pixels // 2
    reciprocal_radius = reciprocal_angstrom_per_pixel*(half_pattern_size - 1)
    simulated_gaussian_sigma = 0.04

    # structure_rotation = euler2mat(np.deg2rad(315), np.deg2rad(90), np.deg2rad(15), axes='rzxz')
    # structure = structure_zb
    structure_rotation = euler2mat(np.deg2rad(119.55382218147206), np.deg2rad(88.986622477291), np.deg2rad(-40.05009628043716), axes='rzxz')
    structure = structure_wz

    lattice_rotated = diffpy.structure.lattice.Lattice(
            structure.lattice.a,
            structure.lattice.b,
            structure.lattice.c,
            structure.lattice.alpha,
            structure.lattice.beta,
            structure.lattice.gamma,
            baserot=structure_rotation)
    structure.placeInLattice(lattice_rotated)

    simulated = diffraction_generator.calculate_ed_data(structure, reciprocal_radius, with_direct_beam=False).as_signal(target_pattern_dimension_pixels, simulated_gaussian_sigma, reciprocal_radius).data
    plt.imshow(simulated)
    plt.show()
    exit(0)


# TODO(simonhog): A lot of this comes from the sped_nn_recognition codebase
def generate_diffraction_library(parameters, phase_names, list_type='complete'):
    if list_type == 'complete':
        cache_file = 'C:/Users/simho/OneDrive/Skydok/MTNANO/Prosjektoppgave/Data/Tmp/library_cache.pickle'
    elif list_type == 'dir':
        cache_file = 'C:/Users/simho/OneDrive/Skydok/MTNANO/Prosjektoppgave/Data/Tmp/library_cache_directional.pickle'
    elif list_type == 'fib':
        cache_file = 'C:/Users/simho/OneDrive/Skydok/MTNANO/Prosjektoppgave/Data/Tmp/library_cache_fib.pickle'
    cache = os.path.exists(cache_file)
    if cache:
        return load_DiffractionLibrary(cache_file, safety=True)

    # plot_debug(parameters)
    h = parameters['zone_h']
    k = parameters['zone_k']
    l = parameters['zone_l']
    beam_energy_keV = parameters['beam_energy_keV']
    specimen_thickness = parameters['specimen_thickness']
    target_pattern_dimension_pixels = parameters['target_pattern_dimension_pixels']
    reciprocal_angstrom_per_pixel = parameters['reciprocal_angstrom_per_pixel']
    # TODO(simonhog): Parameterize
    rotation_list_resolution = np.deg2rad(1)
    max_theta = np.deg2rad(5)
    # TODO(simonhog): Generalize to use (arrays) from parameter file
    # TODO(simonhog): Figure out how diffpy actually want absolute paths on Windows
    structure_zb = diffpy.structure.loadStructure('file:///' + parameters['structure_zb_file'])
    structure_wz = diffpy.structure.loadStructure('file:///' + parameters['structure_wz_file'])

    if list_type == 'complete':
        rotation_list_ZB = generate_complete_rotation_list_euler(
                structure_zb,
                (0, 0, 1),
                (1, 0, 1),
                (1, 1, 1),
                rotation_list_resolution,
                np.deg2rad((84.6, 15)))  # TODO(simonhog): Parameterize
        rotation_list_WZ = generate_complete_rotation_list_euler(
                structure_zb, # TODO(simonhog): Is this correct when I convert the directions also?
                direction_to_cartesian(structure_wz, uvtw_to_uvw(0, 0, 0, 1)),
                direction_to_cartesian(structure_wz, uvtw_to_uvw(1, 1, -2, 0)),
                direction_to_cartesian(structure_wz, uvtw_to_uvw(1, 0, -1, 0)),
                rotation_list_resolution,
                np.deg2rad((111,)))  # TODO(simonhog): Parameterize
    elif list_type == 'dir':
        rotation_list_ZB = generate_rotation_list_euler(structure_zb, h, k, l, max_theta, rotation_list_resolution)
        rotation_list_WZ = generate_rotation_list_euler(structure_wz, h, k, l, max_theta, rotation_list_resolution)
    elif list_type == 'fib':
        rotation_list_ZB = generate_fibonacci_spiral_euler(structure_zb, h, k, l, max_theta, rotation_list_resolution)
        rotation_list_WZ = generate_fibonacci_spiral_euler(structure_wz, h, k, l, max_theta, rotation_list_resolution)


    # rotation_list_ZB = [(90, 44.5, -75), (0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 4, 0), (0, 5, 0)]
    # rotation_list_WZ = [(0.0 , 0.0 , -11.12433428), (0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 4, 0), (0, 5, 0)]
    structure_library = StructureLibrary(
            phase_names,
            [structure_zb, structure_wz],
            [rotation_list_ZB, rotation_list_WZ])
    # phase_names = phase_names[0]
    # structure_library = StructureLibrary(
            # phase_names,
            # [structure_zb],
            # [rotation_list_ZB])


    half_pattern_size = target_pattern_dimension_pixels // 2
    reciprocal_radius = reciprocal_angstrom_per_pixel*(half_pattern_size - 1)
    diffraction_generator = DiffractionGenerator(beam_energy_keV, max_excitation_error = 1/specimen_thickness)
    library_generator = DiffractionLibraryGenerator(diffraction_generator)
    diffraction_library = library_generator.get_diffraction_library(
            structure_library,
            calibration=reciprocal_angstrom_per_pixel,
            reciprocal_radius=reciprocal_radius,
            half_shape=(half_pattern_size, half_pattern_size),
            with_direct_beam=False)
    diffraction_library.pickle_library(cache_file)
    return diffraction_library


# TODO(simonhog): Temporary, waiting for https://github.com/pyxem/pyxem/pull/293
def _euler2axangle_signal(euler):
    """ Find the magnitude of a rotation"""
    return np.array(euler2axangle(euler[0], euler[1], euler[2])[1])


def get_orientation_map(crystal_map):
    """Obtain an orientation image of the rotational angle associated with
    the crystal orientation at each navigation position.
    Returns
    -------
    orientation_map : Signal2D
        The rotation angle assocaiated with the orientation at each
        navigation position.
    """
    eulers = crystal_map.isig[1:4]
    eulers.map(_euler2axangle_signal, inplace=True, show_progressbar=False)
    orientation_map = eulers.as_signal2D((0,1))
    #Set calibration to same as signal
    x = orientation_map.axes_manager.signal_axes[0]
    y = orientation_map.axes_manager.signal_axes[1]
    x.name = 'x'
    x.scale = crystal_map.axes_manager.navigation_axes[0].scale
    x.units = 'nm'
    y.name = 'y'
    y.scale = crystal_map.axes_manager.navigation_axes[0].scale
    y.units = 'nm'
    return orientation_map
