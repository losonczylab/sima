# The following code was copied/adapted from CellProfiler.
#
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2014 Broad Institute
#
# Please see the AUTHORS file for credits.
#
# Website: http://www.cellprofiler.org

import warnings

import numpy as np
try:
    from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
    from pyfftw.interfaces.numpy_fft import rfftn, irfftn
except ImportError:
    from scipy.fftpack import fft2, ifft2
    from numpy.fft import rfftn, irfftn
try:
    from bottleneck import nanmean
except ImportError:
    from scipy import nanmean
import scipy.ndimage as scind
import scipy.sparse


def cross_correlation_3d(pixels1, pixels2):
    '''Align the second image with the first using max cross-correlation

    returns the z,y,x offsets to add to image1's indexes to align it with
    image2

    Many of the ideas here are based on the paper, "Fast Normalized
    Cross-Correlation" by J.P. Lewis
    (http://www.idiom.com/~zilla/Papers/nvisionInterface/nip.html)
    which is frequently cited when addressing this problem.
    '''

    s = np.maximum(pixels1.shape, pixels2.shape)
    fshape = s*2
    #
    # Calculate the # of pixels at a particular point
    #
    i,j,k = np.mgrid[-s[0]:s[0], -s[1]:s[1], -s[2]:s[2] ]
    unit = np.abs(i*j*k).astype(float)
    unit[unit<1]=1 # keeps from dividing by zero in some places
    #
    # Normalize the pixel values around zero which does not affect the
    # correlation, keeps some of the sums of multiplications from
    # losing precision and precomputes t(x-u,y-v) - t_mean
    #
    pixels1 = np.nan_to_num(pixels1-nanmean(pixels1))
    pixels2 = np.nan_to_num(pixels2-nanmean(pixels2))
    #
    # Lewis uses an image, f and a template t. He derives a normalized
    # cross correlation, ncc(u,v) =
    # sum((f(x,y)-f_mean(u,v))*(t(x-u,y-v)-t_mean),x,y) /
    # sqrt(sum((f(x,y)-f_mean(u,v))**2,x,y) * (sum((t(x-u,y-v)-t_mean)**2,x,y)
    #
    # From here, he finds that the numerator term, f_mean(u,v)*(t...) is zero
    # leaving f(x,y)*(t(x-u,y-v)-t_mean) which is a convolution of f
    # by t-t_mean.
    #
    fp1 = rfftn(pixels1.astype('float32'), fshape, axes=(0, 1, 2))
    fp2 = rfftn(pixels2.astype('float32'), fshape, axes=(0, 1, 2))
    corr12 = irfftn(fp1 * fp2.conj(), axes=(0, 1, 2)).real
    #
    # Use the trick of Lewis here - compute the cumulative sums
    # in a fashion that accounts for the parts that are off the
    # edge of the template.
    #
    # We do this in quadrants:
    # q0 q1
    # q2 q3
    # For the first,
    # q0 is the sum over pixels1[i:,j:] - sum i,j backwards
    # q1 is the sum over pixels1[i:,:j] - sum i backwards, j forwards
    # q2 is the sum over pixels1[:i,j:] - sum i forwards, j backwards
    # q3 is the sum over pixels1[:i,:j] - sum i,j forwards
    #
    # The second is done as above but reflected lr and ud
    #
    def get_cumsums(im, fshape):
        im_si = im.shape[0]
        im_sj = im.shape[1]
        im_sk = im.shape[2]
        im_sum = np.zeros(fshape)
        im_sum[:im_si,:im_sj,:im_sk] = cumsum_quadrant(im, False, False, False)
        im_sum[:im_si,:im_sj,-im_sk:] = cumsum_quadrant(im, False, False, True)
        im_sum[:im_si,-im_sj:,:im_sk] = cumsum_quadrant(im, False, True, True)
        im_sum[:im_si,-im_sj:,-im_sk:] = cumsum_quadrant(im, False, True, False)
        im_sum[-im_si:,:im_sj,:im_sk] = cumsum_quadrant(im, True, False, True)
        im_sum[-im_si:,:im_sj,-im_sk:] = cumsum_quadrant(im, True, False, False)
        im_sum[-im_si:,-im_sj:,:im_sk] = cumsum_quadrant(im, True, True, True)
        im_sum[-im_si:,-im_sj:,-im_sk:] = cumsum_quadrant(im, True, True, False)
        #
        # Divide the sum over the # of elements summed-over
        #
        return im_sum / unit

    p1_mean = get_cumsums(pixels1, fshape)
    p2_mean = get_cumsums(pixels2, fshape)
    #
    # Once we have the means for u,v, we can caluclate the
    # variance-like parts of the equation. We have to multiply
    # the mean^2 by the # of elements being summed-over
    # to account for the mean being summed that many times.
    #
    p1sd = np.sum(pixels1**2) - p1_mean**2 * np.product(s)
    p2sd = np.sum(pixels2**2) - p2_mean**2 * np.product(s)
    #
    # There's always chance of roundoff error for a zero value
    # resulting in a negative sd, so limit the sds here
    #
    sd = np.sqrt(np.maximum(p1sd * p2sd, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corrnorm = corr12 / sd
    #
    # There's not much information for points where the standard
    # deviation is less than 1/100 of the maximum. We exclude these
    # from consideration.
    #
    corrnorm[(unit < np.product(s) / 2) &
             (sd < np.mean(sd) / 100)] = 0
    # Also exclude possibilites with few observed pixels.
    corrnorm[unit < np.product(s) / 4] = 0

    return corrnorm


def cross_correlation_2d(pixels1, pixels2):
    '''Align the second image with the first using max cross-correlation

    returns the x,y offsets to add to image1's indexes to align it with
    image2

    Many of the ideas here are based on the paper, "Fast Normalized
    Cross-Correlation" by J.P. Lewis
    (http://www.idiom.com/~zilla/Papers/nvisionInterface/nip.html)
    which is frequently cited when addressing this problem.
    '''
    #
    # We double the size of the image to get a field of zeros
    # for the parts of one image that don't overlap the displaced
    # second image.
    #
    # Since we're going into the frequency domain, if the images are of
    # different sizes, we can make the FFT shape large enough to capture
    # the period of the largest image - the smaller just will have zero
    # amplitude at that frequency.
    #
    s = np.maximum(pixels1.shape, pixels2.shape)
    fshape = s*2
    #
    # Calculate the # of pixels at a particular point
    #
    i,j = np.mgrid[-s[0]:s[0],
                   -s[1]:s[1]]
    unit = np.abs(i*j).astype(float)
    unit[unit<1]=1 # keeps from dividing by zero in some places
    #
    # Normalize the pixel values around zero which does not affect the
    # correlation, keeps some of the sums of multiplications from
    # losing precision and precomputes t(x-u,y-v) - t_mean
    #
    pixels1 = np.nan_to_num(pixels1-nanmean(pixels1))
    pixels2 = np.nan_to_num(pixels2-nanmean(pixels2))
    #
    # Lewis uses an image, f and a template t. He derives a normalized
    # cross correlation, ncc(u,v) =
    # sum((f(x,y)-f_mean(u,v))*(t(x-u,y-v)-t_mean),x,y) /
    # sqrt(sum((f(x,y)-f_mean(u,v))**2,x,y) * (sum((t(x-u,y-v)-t_mean)**2,x,y)
    #
    # From here, he finds that the numerator term, f_mean(u,v)*(t...) is zero
    # leaving f(x,y)*(t(x-u,y-v)-t_mean) which is a convolution of f
    # by t-t_mean.
    #
    fp1 = fft2(pixels1.astype('float32'),fshape)
    fp2 = fft2(pixels2.astype('float32'),fshape)
    corr12 = ifft2(fp1 * fp2.conj()).real

    #
    # Use the trick of Lewis here - compute the cumulative sums
    # in a fashion that accounts for the parts that are off the
    # edge of the template.
    #
    # We do this in quadrants:
    # q0 q1
    # q2 q3
    # For the first,
    # q0 is the sum over pixels1[i:,j:] - sum i,j backwards
    # q1 is the sum over pixels1[i:,:j] - sum i backwards, j forwards
    # q2 is the sum over pixels1[:i,j:] - sum i forwards, j backwards
    # q3 is the sum over pixels1[:i,:j] - sum i,j forwards
    #
    # The second is done as above but reflected lr and ud
    #
    p1_si = pixels1.shape[0]
    p1_sj = pixels1.shape[1]
    p1_sum = np.zeros(fshape)
    p1_sum[:p1_si,:p1_sj] = cumsum_quadrant(pixels1, False, False)
    p1_sum[:p1_si,-p1_sj:] = cumsum_quadrant(pixels1, False, True)
    p1_sum[-p1_si:,:p1_sj] = cumsum_quadrant(pixels1, True, False)
    p1_sum[-p1_si:,-p1_sj:] = cumsum_quadrant(pixels1, True, True)
    #
    # Divide the sum over the # of elements summed-over
    #
    p1_mean = p1_sum / unit

    p2_si = pixels2.shape[0]
    p2_sj = pixels2.shape[1]
    p2_sum = np.zeros(fshape)
    p2_sum[:p2_si,:p2_sj] = cumsum_quadrant(pixels2, False, False)
    p2_sum[:p2_si,-p2_sj:] = cumsum_quadrant(pixels2, False, True)
    p2_sum[-p2_si:,:p2_sj] = cumsum_quadrant(pixels2, True, False)
    p2_sum[-p2_si:,-p2_sj:] = cumsum_quadrant(pixels2, True, True)
    p2_sum = np.fliplr(np.flipud(p2_sum))
    p2_mean = p2_sum / unit
    #
    # Once we have the means for u,v, we can caluclate the
    # variance-like parts of the equation. We have to multiply
    # the mean^2 by the # of elements being summed-over
    # to account for the mean being summed that many times.
    #
    p1sd = np.sum(pixels1**2) - p1_mean**2 * np.product(s)
    p2sd = np.sum(pixels2**2) - p2_mean**2 * np.product(s)
    #
    # There's always chance of roundoff error for a zero value
    # resulting in a negative sd, so limit the sds here
    #
    sd = np.sqrt(np.maximum(p1sd * p2sd, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corrnorm = corr12 / sd
    #
    # There's not much information for points where the standard
    # deviation is less than 1/100 of the maximum. We exclude these
    # from consideration.
    #
    corrnorm[(unit < np.product(s) / 2) &
             (sd < np.mean(sd) / 100)] = 0
    # Also exclude possibilites with few observed pixels.
    corrnorm[unit < np.product(s) / 4] = 0
    return corrnorm


def align_cross_correlation(pixels1, pixels2, displacement_bounds=None):
    '''Align the second image with the first using max cross-correlation

    returns the y,x offsets to add to image1's indexes to align it with
    image2, as well as the correlation at that offset

    Many of the ideas here are based on the paper, "Fast Normalized
    Cross-Correlation" by J.P. Lewis
    (http://www.idiom.com/~zilla/Papers/nvisionInterface/nip.html)
    which is frequently cited when addressing this problem.
    '''
    s = np.maximum(pixels1.shape[:-1], pixels2.shape[:-1])
    fshape = s*2
    if len(s) == 2:
        corr = cross_correlation_2d
    elif len(s) == 3:
        corr = cross_correlation_3d
    else:
        raise ValueError

    corrnorm = sum(corr(pixels1[..., c], pixels2[..., c])
                   for c in range(pixels1.shape[-1])) / pixels1.shape[-1]
    offset = fshape - np.array(pixels1.shape[:-1])
    for i in range(corrnorm.ndim):
        corrnorm = np.roll(corrnorm, offset[i], axis=i)

    if displacement_bounds is not None:
        idx_bounds = displacement_bounds + offset
        corrnorm[:idx_bounds[0][0]] = -np.Inf
        corrnorm[idx_bounds[1][0]:] = -np.Inf
        corrnorm[:, :idx_bounds[0][1]] = -np.Inf
        corrnorm[:, idx_bounds[1][1]:] = -np.Inf
        if idx_bounds.shape[1] == 3:
            corrnorm[:, :, :idx_bounds[0][2]] = -np.Inf
            corrnorm[:, :, idx_bounds[1][2]:] = -np.Inf

    idx = np.unravel_index(np.argmax(corrnorm), fshape)
    return np.array(idx) - offset, corrnorm[idx]


def align_mutual_information(pixels1, pixels2, mask1, mask2):
    '''Align the second image with the first using mutual information

    returns the x,y offsets to add to image1's indexes to align it with
    image2

    The algorithm computes the mutual information content of the two
    images, offset by one in each direction (including diagonal) and
    then picks the direction in which there is the most mutual information.
    From there, it tries all offsets again and so on until it reaches
    a local maximum.
    '''
    def mutualinf(x, y, maskx, masky):
        x = x[maskx & masky]
        y = y[maskx & masky]
        return entropy(x) + entropy(y) - entropy2(x,y)

    maxshape = np.maximum(pixels1.shape, pixels2.shape)
    pixels1 = reshape_image(pixels1, maxshape)
    pixels2 = reshape_image(pixels2, maxshape)
    mask1 = reshape_image(mask1, maxshape)
    mask2 = reshape_image(mask2, maxshape)

    best = mutualinf(pixels1, pixels2, mask1, mask2)
    i = 0
    j = 0
    while True:
        last_i = i
        last_j = j
        for new_i in range(last_i-1,last_i+2):
            for new_j in range(last_j-1, last_j+2):
                if new_i == 0 and new_j == 0:
                    continue
                p2, p1 = offset_slice(pixels2,pixels1, new_i, new_j)
                m2, m1 = offset_slice(mask2, mask1, new_i, new_j)
                info = mutualinf(p1, p2, m1, m2)
                if info > best:
                    best = info
                    i = new_i
                    j = new_j
        if i == last_i and j == last_j:
            return j,i

def offset_slice(pixels1, pixels2, i, j):
    '''Return two sliced arrays where the first slice is offset by i,j
    relative to the second slice.

    '''
    if i < 0:
        height = min(pixels1.shape[0] + i, pixels2.shape[0])
        p1_imin = -i
        p2_imin = 0
    else:
        height = min(pixels1.shape[0], pixels2.shape[0] - i)
        p1_imin = 0
        p2_imin = i
    p1_imax = p1_imin + height
    p2_imax = p2_imin + height
    if j < 0:
        width = min(pixels1.shape[1] + j, pixels2.shape[1])
        p1_jmin = -j
        p2_jmin = 0
    else:
        width = min(pixels1.shape[1], pixels2.shape[1] - j)
        p1_jmin = 0
        p2_jmin = j
    p1_jmax = p1_jmin + width
    p2_jmax = p2_jmin + width

    p1 = pixels1[p1_imin:p1_imax,p1_jmin:p1_jmax]
    p2 = pixels2[p2_imin:p2_imax,p2_jmin:p2_jmax]
    return (p1,p2)

def cumsum_quadrant(x, i_forwards, j_forwards, k_forwards=None):
    '''Return the cumulative sum going in the i, then j direction

    x - the matrix to be summed
    i_forwards - sum from 0 to end in the i direction if true
    j_forwards - sum from 0 to end in the j direction if true
    '''
    if i_forwards:
        x=x.cumsum(0)
    else:
        x=np.flipud(np.flipud(x).cumsum(0))
    if j_forwards:
        x = x.cumsum(1)
    else:
        x =  np.fliplr(np.fliplr(x).cumsum(1))
    if k_forwards is None:
        return x
    if k_forwards:
        return x.cumsum(2)
    else:
        return x[:, :, ::-1].cumsum(2)[:, :, ::-1]

def entropy(x):
    '''The entropy of x as if x is a probability distribution'''
    histogram = scind.histogram(x.astype(float), np.min(x), np.max(x), 256)
    n = np.sum(histogram)
    if n > 0 and np.max(histogram) > 0:
        histogram = histogram[histogram!=0]
        return np.log2(n) - np.sum(histogram * np.log2(histogram))/n
    else:
        return 0

def entropy2(x,y):
    '''Joint entropy of paired samples X and Y'''
    #
    # Bin each image into 256 gray levels
    #
    x = (stretch(x) * 255).astype(int)
    y = (stretch(y) * 255).astype(int)
    #
    # create an image where each pixel with the same X & Y gets
    # the same value
    #
    xy = 256*x+y
    xy = xy.flatten()
    sparse = scipy.sparse.coo_matrix((np.ones(xy.shape),
                                      (xy,np.zeros(xy.shape))))
    histogram = sparse.toarray()
    n=np.sum(histogram)
    if n > 0 and np.max(histogram) > 0:
        histogram = histogram[histogram>0]
        return np.log2(n) - np.sum(histogram * np.log2(histogram)) / n
    else:
        return 0

def reshape_image(source, new_shape):
    '''Reshape an image to a larger shape, padding with zeros'''
    if tuple(source.shape) == tuple(new_shape):
        return source

    result = np.zeros(new_shape, source.dtype)
    result[:source.shape[0], :source.shape[1]] = source
    return result

def stretch(image, mask=None):
    '''Normalize an image to make the minimum zero and maximum one

    image - pixel data to be normalized
    mask  - optional mask of relevant pixels. None = don't mask

    returns the stretched image
    '''
    image = np.array(image, float)
    if np.product(image.shape) == 0:
        return image
    if mask is None:
        minval = np.min(image)
        maxval = np.max(image)
        if minval == maxval:
            if minval < 0:
                return np.zeros_like(image)
            elif minval > 1:
                return np.ones_like(image)
            return image
        else:
            return (image - minval) / (maxval - minval)
    else:
        significant_pixels = image[mask]
        if significant_pixels.size == 0:
            return image
        minval = np.min(significant_pixels)
        maxval = np.max(significant_pixels)
        if minval == maxval:
            transformed_image = minval
        else:
            transformed_image = ((significant_pixels - minval) /
                                 (maxval - minval))
        result = image.copy()
        image[mask] = transformed_image
        return image

