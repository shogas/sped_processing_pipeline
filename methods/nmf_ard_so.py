import numpy as np

from scipy.special import gammaln
import hyperspy.api as hs
from hyperspy.learn.mva import LearningResults

# TODO(simonhog): Implement in hyperspy directly?
# TODO(simonhog): Consistent naming with hyperspy, this is from the original ARD_SO_NMF implementation
@profile
def decomposition_ard_so(
        s,
        n_components,
        wo=0.1,
        n_reps=3,
        max_iterations=100,
        alpha=1+1e-15,
        threshold_merge=0.99,
        random_seed=0,
        flag_nonnegative=True):
    # NOTE: Started with implementation from https://github.com/MotokiShiga/stem-nmf/blob/master/python/libnmf.py

    #       Motoki Shiga, Kazuyoshi Tatsumi, Shunsuke Muto, Koji Tsuda,
    #       Yuta Yamamoto, Toshiyuki Mori, Takayoshi Tanji,
    #       "Sparse Modeling of EELS and EDX Spectral Imaging Data by Nonnegative Matrix Factorization",
    #       Ultramicroscopy, Vol.170, p.43-59, 2016.
    #

    X = s.data.reshape(s.axes_manager.navigation_size, s.axes_manager.signal_size)

    eps = np.finfo(X.dtype).eps

    n_X_xy, n_X_channels = X.shape
    X_mean = np.mean(X)
    beta = X_mean * (alpha - 1) * np.sqrt(n_X_channels) / n_components
    # TODO(simonhog): Descriptive name from article
    const = n_components * (gammaln(alpha) - alpha * np.log(beta))
    np.random.seed(random_seed)

    obj_best = np.array([np.inf])
    for rep in range(n_reps):
        print('Repetition ' + str(rep))

        # Initialization
        # TODO(simonhog): Get initialization code from sklearn/hyperspy
        C = (np.random.rand(n_X_xy, n_components) + 1) * (np.sqrt(X_mean / n_components))
        L = (np.sum(C, axis=0) + beta) / (n_X_channels + alpha + 1)
        cj = np.sum(C, axis=1)
        i = np.random.choice(n_X_xy, n_components)
        S = X[i, :].T
        for j in range(n_components):
            c = np.sqrt(S[:, j].T @ S[:, j])
            if c > 0:
                S[:, j] = S[:, j] / c
            else:
                S[:, j] = 1 / np.sqrt(n_X_channels)

        X_est = C @ S.T
        variance = np.mean((X - X_est)**2)
        obj = np.zeros(max_iterations)
        lambdas = np.zeros((max_iterations, n_components))

        # Iteration
        for itr in range(max_iterations):
            print('  Iteration ' + str(itr))
            # Update S
            XC = X.T @ C
            C_squared = C.T @ C
            for j in range(n_components):
                S[:, j] = XC[:, j] - S @ C_squared[:, j] + C_squared[j, j] * S[:, j]
                if flag_nonnegative:
                    S[:, j] = (S[:, j] + np.abs(S[:, j])) / 2
                c = np.sqrt(S[:, j].T @ S[:, j])
                if c > 0:
                    S[:, j] = S[:, j] / c
                else:
                    S[:, j] = 1 / np.sqrt(n_X_channels)

            # Update C
            XS = X @ S
            S_squared = S.T @ S
            for j in range(n_components):
                cj -= C[:, j]
                C[:, j] = XS[:, j] - C @ S_squared[:, j] + S_squared[j, j] * C[:, j]
                C[:, j] -= variance / L[j]
                if wo > 0:  # TODO(simonhog): Never used if wo < 0. Error before starting instead. (Or not. This allows pure ARD instead of ARD-SO)
                    C[:, j] -= wo * (cj.T @ C[:, j]) / (cj.T @cj) * cj
                C[:, j] = (C[:, j] + np.abs(C[:, j])) / 2  # Clamp bottom 0. TODO(simonhog): Is this quicker than np.min(0, C[:, j])?
                cj += C[:, j]

            # Merge components if their spectra are almost the same
            if itr > 3:
                SS = S.T @ S
                i, j = np.where(SS >= threshold_merge)
                m = i < j
                i = i[m]
                j = j[m]
                for n in range(len(i)):
                    S[:, j[n]] = 1 / np.sqrt(n_X_channels)
                    C[:, i[n]] = np.sum(C[:, np.r_[i[n], j[n]]], axis=1)
                    C[:, j[n]] = 0
            if np.sum(cj) < eps:
                C[:, :] = eps

            # Update lambda (ARD parameters)
            L = (np.sum(C, axis=0) + beta) / (n_X_xy + alpha + 1) + eps
            lambdas[itr, :] = L.copy()

            # Update variance of additive Gaussian noise)
            X_est = C @ S.T
            print('    X_est shape: {}'.format(X_est.shape))
            asdf = (X - X_est)**2
            variance = np.mean(asdf)

            # Object function (negative log likelihood)
            obj[itr] = n_X_xy * n_X_channels / 2 * np.log(2 * np.pi * variance) + n_X_xy * n_X_channels / 2  # MSE
            obj[itr] += (L ** (-1)).T @ (np.sum(C, axis=0) + beta).T + (n_X_xy + alpha + 1) * np.sum(np.log(L), axis=0) + const

            # TODO(simonhog): Original implementation uses & (binary and) instead of and?
            if itr > 1 and np.abs(obj[itr - 1] - obj[itr]) < 1e-10:  # TODO(simonhog): Adjustable convergence parameter?
                obj = obj[0:itr]
                lambdas = lambdas[0:itr, :].copy()
                print('Stopping after iteration {} because object function difference < 1e-10'.format(itr))
                break

        # Choose the best result
        if obj_best[-1] > obj[-1]:
            print('  Updating best on repetition ' + str(rep))
            obj_best = obj.copy()
            C_best = C.copy()
            S_best = S.copy()
            lambdas_best = lambdas.copy()

    obj_fun = obj_best
    C_best[C_best < eps] = 0
    S_best[S_best < eps] = 0
    L_best = (np.sum(C, axis=0) + beta) / (n_X_xy + alpha + 1)
    k = np.argsort(-L_best)
    n_comp_best = np.sum(L_best[k] > eps)
    ks = k[:n_comp_best]

    # TODO(simonhog): Return values, add to signal
    L = L_best[ks]
    lambdas = lambdas_best[:, k]
    X_est = C @ S.T
    variance = np.mean((X - X_est)**2)

    if not hasattr(s, 'learning_results'):
        s.learning_results = LearningResults()

    s.learning_results.factors = S_best[:, ks]
    s.learning_results.loadings = C_best[:, ks]
    s.learning_results.decomposition_algorithm = 'ARD_SO_NMF'
    s.learning_results.output_dimensions = n_comp_best
    print('Found {} components'.format(n_comp_best))


def factorize(diffraction_patterns, parameters):
    s = hs.signals.Signal2D(diffraction_patterns)

    factor_count = 5
    decomposition_ard_so(s, n_components=factor_count, n_reps=2, max_iterations=100, wo=0.05)

    if s.learning_results.output_dimensions > 0:
        factors = s.get_decomposition_factors().data
        loadings = s.get_decomposition_loadings().data
    else:
        factors = np.array([diffraction_patterns[0, 0]])
        loadings = np.ones((1, diffraction_patterns.data.shape[0], diffraction_patterns.data.shape[1]))

    scale = loadings.max()
    factors *= scale
    loadings /= scale
    return factors, loadings

