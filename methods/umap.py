from datetime import datetime
import os

import numpy as np
import umap
import hdbscan

def process(diffraction_patterns, parameters):
    """Factorizes diffraction patterns by dimensionality reduction using UMAP
    followed by clustering using HDBSCAN.

    Parameters
    ---------
    diffraction_patterns : numpy.ndarray
        4D numpy array containing the diffraction patterns.
    parameters : dict
        Dictionary of parameters:
            'output_dir_run' : string
                Output directory for storing UMAP results.
            'umap_neighbors' : int
                Number of nearest neighbors to check.
                See https://umap-learn.readthedocs.io/en/latest/parameters.html#n-neighbors
            'umap_min_dist' : float
                Minimum distance, typically 0.0 for clustering.
                See https://umap-learn.readthedocs.io/en/latest/parameters.html#min-dist
            'umap_n_components' : int
                Number of dimensions to reduce into before clustering.
            'umap_random_seed' : int
                Random seed for reproducability. Optional.
            'umap_cluster_size' : int
                Smallest grouping to consider a cluster.
                See https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-min-cluster-size
            'umap_cluster_min_samples' : int
                How conservative the clustering is. Larger numbers assigns more
                points as noise.
                See https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-min-samples

    Returns
    -------
    results : (factors, loadings)
        Tuple of factors and corresponding loadings, where the factors are the
        weighted average of each cluster and the loading maps contains cluster
        membership strength to the corresponding cluster.
    result_type : string
        decomposition
    """
    # Read the input parameters
    nav_width, nav_height, signal_width, signal_height = diffraction_patterns.shape
    random_seed = parameters['umap_random_seed'] if 'umap_random_seed' in parameters else None

    # Reshape to a two-dimensional matrix, one row per diffraction pattern,
    # as required by UMAP
    data_flat = diffraction_patterns.reshape(-1, signal_width*signal_height)

    # Do the projection to a lower dimensional space (given by 'n_components')
    # using UMAP with the parameters specified above.
    embedding = umap.UMAP(
        n_neighbors =parameters['umap_neighbors'],
        min_dist    =parameters['umap_min_dist'],
        n_components=parameters['umap_n_components'],
        random_state=random_seed,
    ).fit_transform(data_flat)

    # Save the results from UMAP, since this is the most expensive step.
    # Clustering can then be done separately to adjust parameters.
    save_embedding(embedding, parameters['output_dir_run'])

    # Cluster the low-dimensional data using HDBSCAN and the parameters
    # specified above.
    clusterer = hdbscan.HDBSCAN(
        min_samples=parameters['umap_cluster_min_samples'],
        min_cluster_size=parameters['umap_cluster_size'],
    ).fit(embedding)

    # Format the return values
    label_count = clusterer.labels_.max() + 1  # include 0
    if label_count <= 0:
        # No clusters, return all zeros
        factors = np.zeros((1, signal_width, signal_height))
        loadings = np.zeros((1, nav_width, nav_height))
    else:
        # Allocate space for the results
        factors = np.empty((label_count, signal_width, signal_height))
        loadings = np.empty((label_count, nav_width, nav_height))

        for label in range(label_count):
            # Set the loading from all the HDBSCAN probabilities,
            loadings[label] = clusterer.probabilities_.reshape(nav_width, nav_height)
            # but mask out the results not matching this label
            mask = (clusterer.labels_ == label).reshape(nav_width, nav_height)
            loadings[label][~mask] = 0.0
            # Calculate factors as a weighted average of cluster members
            # and reshape to the correct shape
            factors[label] = np.average(
                data_flat,
                weights=loadings[label].ravel(),
                axis=0).reshape(signal_width, signal_height)

    return (factors, loadings), 'decomposition'


def save_embedding(embedding, output_dir):
    """Save the embedding with a timestamp

    Parameters
    ----------
    embedding : numpy.ndarray
        Numpy array containing the embedding
    output_dir : string
        Path to directory in which the embedding should be saved
    """
    unique_identifier = datetime.now().strftime('%Y%m%d_%H_%M_%S_%f')
    path = os.path.join(
        output_dir,
        'embedding_{}.npy'.format(unique_identifier))
    np.save(path, embedding)
