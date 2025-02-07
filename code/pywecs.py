import numpy as np
from skimage.filters import threshold_otsu
from skimage.restoration import denoise_wavelet, cycle_spin
from sklearn.cluster import KMeans

def denoise_swt(img, wavelet='haar', J=2):
    denoise_kwargs = dict(
        channel_axis=None, wavelet=wavelet, rescale_sigma=True
    )
    
    denoised_img = cycle_spin(
        img,
        func=denoise_wavelet,
        max_shifts=J,
        func_kw=denoise_kwargs,
        channel_axis=None,
        num_workers=1
    )

    return denoised_img

def ECS(x, time_axis=0, smooth_x=None):
    
    assert len(x.shape) == 3, "'x' is not three-dimensional"
    mean_image = x.mean(axis=time_axis)
    
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

def apply_wavelet(x, show_progress=True):
    xwav = np.ndarray(x.shape)
    t = xwav.shape[0]
    for i in range(t):
        xwav[i, :, :] = denoise_swt(x[i, :, :])
        if show_progress:
            print("Applying Wavelet: ", str(i+1), "/", str(t), end="\r")
    return xwav

def WECS(x, print_progress=True):
    x_denoised = apply_wavelet(x, show_progress=print_progress)
    x_wecs = ECS(x, smooth_x=x_denoised)
    return x_wecs

def segment2d(x, method='ot'):
    assert method in ('ot', 'ki', 'km'), 'Method needs to be Otsu (ot), Kittler-Illingworth (ki) or KMeans (km)'
    if method == 'ot':
        th = threshold_otsu(x)
        binary = x >= th
        binary = binary.astype('uint8')
        return binary
    elif method == 'ki':
        th = threshold_ki(x)
        binary = x >= th
        binary = binary.astype('uint8')
        return binary
    elif method == 'km':
        kmeans = KMeans(n_clusters=2, random_state=0).fit(x.reshape(-1,1))
        res = kmeans.cluster_centers_[kmeans.labels_].reshape(x.shape)
        return res
    
# Adapted from https://github.com/al42and/cv_snippets/blob/master/kittler.py
# Copyright (c) 2014, Bob Pepin
# Copyright (c) 2016, Andrey Alekseenko
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the distribution
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

def threshold_ki(img, min_val=0, max_val=1):
    """
    The reimplementation of Kittler-Illingworth Thresholding algorithm by Bob Pepin.
    Original Matlab code: https://www.mathworks.com/matlabcentral/fileexchange/45685-kittler-illingworth-thresholding
    Paper: Kittler, J. & Illingworth, J. Minimum error thresholding. Pattern Recognit. 19, 41–47 (1986).
    """
    h, g = np.histogram(img.ravel(), 265, [min_val, max_val])
    g = g[:-1]
    c = np.cumsum(h)
    m = np.cumsum(h * g)
    s = np.cumsum(h * g**2)
    cb = c[-1] - c
    mb = m[-1] - m
    sb = s[-1] - s
    with np.errstate(invalid='ignore', divide='ignore'):
        sigma_f = np.sqrt(s/c - (m/c)**2)
        sigma_b = np.sqrt(sb/cb - (mb/cb)**2)
        p = c / c[-1]
        v = p * np.log(sigma_f) + (1-p)*np.log(sigma_b) - p*np.log(p) - (1-p)*np.log(1-p)
    v[~np.isfinite(v)] = np.inf
    idx = np.argmin(v)
    t = g[idx]
    return t