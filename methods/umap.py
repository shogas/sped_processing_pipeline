import numpy as np

import hyperspy.api as hs
import umap
import hdbscan
import time

def factorize(diffraction_patterns, parameters):
    nav_width = diffraction_patterns.shape[0]
    nav_height = diffraction_patterns.shape[1]
    signal_width = diffraction_patterns.shape[2]
    signal_height = diffraction_patterns.shape[3]
    data_flat = diffraction_patterns.reshape(-1, signal_width*signal_height)
    random_seed = 42
    start = time.time()
    # TODO(simonhog): PCA for reduction to fewer dimensions (~100) first?
    embedding = umap.UMAP(
        n_neighbors=30,
        min_dist=0.00,
        n_components=2,
        random_state=random_seed,
    ).fit_transform(data_flat)

    start = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_samples=20,
        min_cluster_size=500,
    ).fit(embedding)

    # TODO(simonhog): Template mapping for component -> physical phases
    label_count = clusterer.labels_.max()
    factors = np.empty((label_count, signal_width, signal_height))
    loadings = np.empty((label_count, nav_width, nav_height))
    for label in range(label_count):
        mask = (clusterer.labels_ == label).reshape(nav_width, nav_height)
        loadings[label] = clusterer.probabilities_.reshape(nav_width, nav_height)
        loadings[label][~mask] = 0.0
        factors[label] = np.average(diffraction_patterns.reshape(-1, signal_width*signal_height), weights=loadings[label].ravel(), axis=0).reshape(signal_width, signal_height)
    return factors, loadings

