import numpy as np
from numba import njit
import pywt
from skimage.filters import threshold_otsu, threshold_li

@njit
def ECS(x, smooth_x=None):
    
    assert len(x.shape) == 3, "'x' is not three-dimensional"
    mean_image = nb_mean_axis_0(x)
    
    if smooth_x is not None:
        assert len(smooth_x.shape) == 3, "'smooth_x' is not three-dimensional"
        assert x.shape == smooth_x.shape, "'x' and 'smooth_x' are different shapes"
        cube = smooth_x.astype(np.float32)
    else:
        cube = x

    R = np.empty(mean_image.shape, np.float32)
    D = np.empty(cube.shape, np.float32)

    dims = mean_image.shape
    
    lin = dims[0]
    col = dims[1]
    
    for i in range(0, cube.shape[0]):
        D[i] = (cube[i] - mean_image)**2
    
    d = D.sum(axis=1).sum(axis=1).flatten()

    for i in range(lin):
        for j in range(col):
            R[i, j] = np.abs(np.corrcoef(d, D[:, i, j])[0][1])
    
    return R

@njit
def apply_along_axis_0(func1d, arr):
    """Like calling func1d(arr, axis=0)"""
    if arr.size == 0:
        raise RuntimeError("Must have arr.size > 0")
    ndim = arr.ndim
    if ndim == 0:
        raise RuntimeError("Must have ndim > 0")
    elif 1 == ndim:
        return func1d(arr)
    else:
        result_shape = arr.shape[1:]
        out = np.empty(result_shape, arr.dtype)
        _apply_along_axis_0(func1d, arr, out)
        return out
    
@njit
def _apply_along_axis_0(func1d, arr, out):
    """Like calling func1d(arr, axis=0, out=out). Require arr to be 2d or bigger."""
    ndim = arr.ndim
    if ndim < 2:
        raise RuntimeError("_apply_along_axis_0 requires 2d array or bigger")
    elif ndim == 2:  # 2-dimensional case
        for i in range(len(out)):
            out[i] = func1d(arr[:, i])
    else:  # higher dimensional case
        for i, out_slice in enumerate(out):
            _apply_along_axis_0(func1d, arr[:, i], out_slice)
    
@njit
def nb_mean_axis_0(arr):
    return apply_along_axis_0(np.mean, arr)

def apply_wavelet(x, show_progress=True):
    xwav = np.ndarray(x.shape)
    t = xwav.shape[0]
    for i in range(t):
        xwav[i, :, :] = pywt.swt2(
            x[i, :, :], 
            wavelet="haar", 
            level=2,
            start_level=0
        )[0][0]
        if show_progress:
            print(f"Applying Wavelet: {str(i+1)}/{str(t)}", end="\r")
    return xwav

def WECS(x):
    smooth_x = apply_wavelet(x)
    res = ECS(x, smooth_x)
    return res

def segment_otsu(x):
    th = threshold_otsu(x)
    binary = x > th
    binary = binary.astype('uint8')
    return binary