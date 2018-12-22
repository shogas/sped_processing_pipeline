import numpy as np

def cepstrum(z):
    z = np.fft.fft2(z)
    z = z**2
    z = np.log1p(np.abs(z))
    z = np.fft.ifft2(z)
    z = np.fft.fftshift(z)
    z = np.abs(z)
    z = z**2
    return z


def cepstrum_power(z):
    """Transform the input using the cepstrum transform.

    Parameters
    ----------
        z : 2D np.ndarray
            Input to be transformed.

    Returns
    -------
        z : 2D np.ndarray
            Transformed input.
    """
    z = np.fft.fft2(z)
    z = z * np.conj(z)
    np.seterr(divide='ignore')
    z = np.log(z)
    np.seterr(divide='warn')
    z = np.fft.ifft2(z)
    z = np.fft.fftshift(z)
    z = z * np.conj(z)
    return z.real


def cepstrum_complex(z):
    """Transform the input using the complex cepstrum transform.

    Parameters
    ----------
        z : 2D np.ndarray
            Input to be transformed.

    Returns
    -------
        z : 2D np.ndarray
            Transformed input.
    """
    z = np.fft.fft2(z)
    np.seterr(divide='ignore')
    z = np.log(z)
    np.seterr(divide='warn')
    z = np.fft.ifft2(z)
    z = np.fft.fftshift(z)
    return z


def cepstrum_real(z):
    z = np.fft.fft2(z)
    z = np.abs(z)
    np.seterr(divide='ignore')
    z = np.log(z)
    np.seterr(divide='warn')
    z = np.fft.ifft2(z)
    z = np.fft.fftshift(z)
    return z


def preprocessor_affine_transform(data, parameters):
    """Apply an affine transform to `data`.

    Parameters
    ----------
    data : numpy.ndarray
        4D numpy array containing the diffraction patterns.
    parameters : dict
        Dictionary of parameters:
        'scale_x' : float
            Diffraction image scaling in x direction.
        'scale_y' : float
            Diffraction image scaling in y direction.
        'offset_x' : float
            Diffraction image offset in x direction.
        'offset_y' : float
            Diffraction image offset in y direction.

    Returns
    -------
    data : numpy.ndarray
        Data after applying the transform specified by paramteters.
    """
    signal = pxm.ElectronDiffraction(data)
    signal.apply_affine_transformation(np.array([
            [parameters['scale_x'], 0, parameters['offset_x']],
            [0, parameters['scale_y'], parameters['offset_y']],
            [0, 0, 1]
        ]))
    return signal.data


def preprocessor_gaussian_difference(data, parameters):
    """Remove background using the Gaussian difference method.

    Parameters
    ----------
    data : numpy.ndarray
        4D numpy array containing the diffraction patterns.
    parameters : dict
        Dictionary of parameters:
        'gaussian_sigma_min' : float
            Standard deviation of the smallest Gaussian blur.
        'gaussian_sigma_max' : float
            Standard deviation of the largest Gaussian blur.

    Returns
    -------
    data : numpy.ndarray
        Data after background removal.
    """
    signal = pxm.ElectronDiffraction(data)
    sig_width = signal.axes_manager.signal_shape[0]
    sig_height = signal.axes_manager.signal_shape[1]

    signal = signal.remove_background(
            'gaussian_difference',
            sigma_min=parameters['gaussian_sigma_min'],
            sigma_max=parameters['gaussian_sigma_max'])
    # Rescale the result to <= 1
    signal.data /= signal.data.max()

    return signal.data


def preprocessor_hdome(data, parameters):
    signal = pxm.ElectronDiffraction(data)
    signal = signal.remove_background('h-dome', h=parameters['hdome_h'])
    signal.data *= 1/signal.data.max()
    return signal.data

