import numpy as np

import hyperspy.api as hs
import umap
import hdbscan

def factorize(diffraction_patterns, parameters):
    nav_width = diffraction_patterns.shape[0]
    nav_height = diffraction_patterns.shape[1]
    signal_width = diffraction_patterns.shape[2]
    signal_height = diffraction_patterns.shape[3]
    data_flat = diffraction_patterns.reshape(-1, signal_width*signal_height)

    embedding = umap.UMAP(
        n_neighbors =parameters['umap_neighbors'],
        min_dist    =parameters['umap_min_dist'],
        n_components=parameters['umap_n_components'],
        random_state=parameters['umap_random_seed'],  # For consistency
    ).fit_transform(data_flat)

    clusterer = hdbscan.HDBSCAN(
        min_samples=parameters['umap_cluster_min_samples'],
        min_cluster_size=parameters['umap_cluster_size'],
    ).fit(embedding)

    # TODO(simonhog): Template mapping for component -> physical phases
    label_count = clusterer.labels_.max()
    if label_count <= 0:
        factors = np.zeros((1, signal_width, signal_height))
        loadings = np.zeros((1, nav_width, nav_height))
    else:
        factors = np.empty((label_count, signal_width, signal_height))
        loadings = np.empty((label_count, nav_width, nav_height))
        for label in range(label_count):
            mask = (clusterer.labels_ == label).reshape(nav_width, nav_height)
            loadings[label] = clusterer.probabilities_.reshape(nav_width, nav_height)
            loadings[label][~mask] = 0.0
            factors[label] = np.average(diffraction_patterns.reshape(-1, signal_width*signal_height), weights=loadings[label].ravel(), axis=0).reshape(signal_width, signal_height)
    return (factors, loadings), 'decomposition'

