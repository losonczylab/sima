from __future__ import absolute_import
from __future__ import division
from builtins import next
from builtins import zip
from builtins import map
from builtins import range
from past.utils import old_div
from builtins import object
import itertools as it
import multiprocessing
import warnings

import numpy as np
try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean
import scipy.ndimage.filters

from . import motion
from sima.misc.align import align_cross_correlation

# define globals for dftreg
# might be better to use global namespace struct? or attributes?
global fftn, ifftn, im_dim, im_dtype, n_workers, verbose_state

# imports for dftreg
from scipy.ndimage.interpolation import shift
from functools import partial
import time


class DiscreteFourier2D(motion.MotionEstimationStrategy):
    """
Motion correction of image sequences by 'efficient subpixel image registration
by cross correlation'. A reference image is iteratively computed by aligning
and averaging a subset of images/frames.
Lloyd Russell 2015 (and Christoph, and Marius (Adam?) for initial MATLAB implementation?)
*******************************************************************************
Implements skimage.feature.register_translation, which is a port of MATLAB code
by Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, "Efficient
subpixel image registration algorithms," Optics Letters 33, 156-158 (2008).

Relating to implementation of skimage.feature.register_translation:
Copyright (C) 2011, the scikit-image team
All rights reserved.
THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

*******************************************************************************
    """

    def __init__(self, upsample_factor=1, num_images_for_mean=100,
             randomise_frames=True, err_thresh=0.02, max_iterations=5,
             use_fftw=False, rotation_scaling=False, save=False,
             save_fmt='mptiff', save_name='none', n_processes=1, verbose=False):
        self._params = dict(locals())
        del self._params['self']

    def _estimate(self, dataset):
        """

        Parameters
        ----------

        Returns
        -------
        shifts : array
            (2, num_frames*num_cycles)-array of integers giving the
            estimated displacement of each frame
        """
        params = self._params

        displacements = []  # a list

        for seq_idx, sequence in enumerate(dataset):
            frame_shape = dataset[seq_idx].frame_shape
            num_planes = frame_shape[0]
            num_channels = frame_shape[3]
            if num_channels > 1:
                raise NotImplementedError("Error: only one colour channel can be used for DFT motion correction")

            for plane_idx in range(num_planes):
                # load into memory... need to pass numpy array to dftreg.
                # could(should?) rework it to instead accept tiff array
                print('Loading...')
                t0 = time.time()
                frames = np.squeeze(sequence[:,plane_idx])  # reshape, only one plane at a time
                e1 = time.time() - t0
                print('Loaded in ' + str(e1) + ' s')

                # do the registering
                # registered_frames return is useless, sima later uses the displacements to shift the
                # image (apply_displacements in sima/sequence.py: _align method of _MotionCorrectedSequence
                # class) but this shifting is only pixel-level, much better results if sub-pixel were possible
                # - replace sima's way of shifting? this may run into problems when sima then crops the final
                # image so no empty rows/columns at edge of any frame in the video (trim_criterion)
                dx,dy,registered_frames = register(
                    frames, upsample_factor=params['upsample_factor'], use_fftw=params['use_fftw'],
                    n_processes=params['n_processes'], verbose=params['verbose'],
                    save=params['save'], save_name=params['save_name'])

                # get results into a shape sima likes...
                frame_shifts = np.zeros([len(frames),num_planes,2])
                for idx,frame in enumerate(sequence):
                    frame_shifts[idx,plane_idx] = [dy[idx], dx[idx]]
            displacements.append(frame_shifts)

        return displacements



def register(input_array, upsample_factor=1, num_images_for_mean=100,
             randomise_frames=True, err_thresh=0.02, max_iterations=5,
             use_fftw=False, rotation_scaling=False, save=False,
             save_fmt='mptiff', save_name='none', n_processes=1, verbose=False):
    """
    Master function. Establish parameters. Make aligned mean image. Register
    each frame in input array to aligned mean image.

    Parameters
    ----------
    input array : np.ndarray
        the frames to align
    upsample : int
        upsample factor. final pixel alignment has resolution of
        1/upsample_factor. if 1 only pixel level shifts are made - faster -
        and no interpolation (optional, default=1)
    num_images_for_mean : int
        number of images to use to make the aligned mean image (optional,
        default=100)
    randomise_frames : bool
        randomise the images selected to make the mean image? if false the
        first 'num_frames_for_mean' frames will be used (optional,
        default=True)
    err_thresh : float
        the error threshold level at which to stop iterating over the mean
        image alignment (optional, default=0.02)
    max_iterations : int
        the maximum number of iterations to compute the aligned mean image
        (optional, default=5)
    use_fftw : bool
        choose whether to use fftw methods (slightly faster) requires PyFFTW3.
        if false, will use numpy methods. (optional, default=False)
    rotation_scaling : bool
        not yet implemented. (optional, default=false)
    save : bool
        choose whether to save the final registered array of images to disk
        from within method (optional, default=False)
    save_name : string
        the filename for saved file (optional, default='none')
    save_fmt : string
        the tiff format to save as. options include 'mptiff', 'bigtiff',
        'singles' (optional, default='mptiff'
    n_processes : int or 'auto'
        number of workers to use (multiprocessing). if 'auto' number of workers
        is number of cpus. (optional, default=1)
    verbose : bool
        enable verbose mode (optional, default:False)

    Returns
    -------
    dx : float array
        horizontal pixel offsets. shift the target image by this amount to
        align with reference
    dy : float array
        vertical pixel offsets. shift the target image by this amount to align
        with reference
    registered_frames : np.ndarray
        the aligned frames
    """

    # start timer
    t0 = time.time()

    # configure settings and get image parameters
    _configure(input_array, use_fftw=use_fftw,
               verbose=verbose, n_processes=n_processes)

    # make a mean image
    mean_img = _make_mean_img(input_array,
                              num_images_for_mean=num_images_for_mean,
                              randomise_frames=randomise_frames,
                              err_thresh=err_thresh,
                              max_iterations=max_iterations,
                              upsample_factor=upsample_factor)
    e1 = time.time() - t0
    if verbose_state:
        print('    Time taken: ' + str(e1) + ' s')

    # register all frames
    dx, dy, registered_frames = _register_all_frames(
        input_array, mean_img, upsample_factor=upsample_factor)
    e2 = time.time() - t0 - e1
    if verbose_state:
        print('    Time taken: ' + str(e2) + ' s')

    # save?
    if save:
        _save_registered_frames(registered_frames, save_name, save_fmt)
        e3 = time.time() - t0 - e1 - e2
        if verbose_state:
            print('    Time taken: ' + str(e3) + ' s')

    total_time = time.time() - t0
    if verbose_state:
        print('Completed in: ' + str(total_time) + ' s')

    return dx, dy, registered_frames


def _configure(input_array, use_fftw=False, verbose=False, n_processes=1):
    """
    Setup. Decide how many workers to use (multiprocessing), which fft methods
    to use and get details of the input images

    Parameters
    ----------
    input_array : array
    use_fftw : bool
    upsample_val : int
    verbose : bool

    Returns
    -------
    None. But sets globals...
    """
    # define globals
    global fftn, ifftn, im_dim, im_dtype, n_workers, verbose_state

    # verbose mode?
    if verbose:
        verbose_state = True
    else:
        verbose_state = False

    # workers
    if n_processes == 'auto':
        n_workers = multiprocessing.cpu_count()
    else:
        n_workers = n_processes
    if verbose_state:
        print('Using ' + str(n_workers) + ' worker(s)')

    # num threads
    nthreads = 1

    # get fft methods
    fftn, ifftn = _get_ffts(use_fftw=use_fftw, nthreads=nthreads)

    # get image details
    im_dim = input_array.shape
    im_dtype = input_array.dtype


def _make_mean_img(input_array, num_images_for_mean=100, randomise_frames=True,
                   err_thresh=0.02, max_iterations=5, upsample_factor=10):
    """
    Make an aligned mean image to use as reference to which all frames are
    later aligned.

    Parameters
    ----------
    num_images_for_mean : int
        how many images are used to make the mean reference image
        (default=100)
    randomise_frames : bool
        randomise the frames used to make the mean image? If False the first
        N images are used (default = True)
    err_thresh : float
        the threshold of mean pixel offset at which to stop aligning the mean
        image (default = 0.02)
    max_iterations : int
        number of maximum iterations, if error threshold is never met
        (default = 5)
    Returns
    -------
    mean_img : np.ndarray (size of input images)
        the final aligned mean image
    """

    if num_images_for_mean > im_dim[0]:
        num_images_for_mean = im_dim[0]

    frames_for_mean = np.zeros([num_images_for_mean, im_dim[1], im_dim[2]], dtype=im_dtype)

    if randomise_frames:
        if verbose_state:
            print('Making aligned mean image from ' +
                  str(num_images_for_mean) + ' random frames...')

        for idx,frame_num in enumerate(np.random.choice(im_dim[0], size=num_images_for_mean, replace=False)):
            frames_for_mean[idx] = input_array[frame_num]
    else:
        if verbose_state:
            print('Making aligned mean image from first ' +
                  str(num_images_for_mean) + ' frames...')
        for idx,frame_num in enumerate(range(num_images_for_mean)):
            frames_for_mean[idx] = input_array[frame_num]

    mean_img = np.mean(frames_for_mean, 0)
    iteration = 1
    mean_img_err = 9999

    while mean_img_err > err_thresh and iteration < max_iterations:  # not final conditions
        # # configure pool of workers (multiprocessing)
        # pool = multiprocessing.Pool(n_workers, maxtasksperchild=1)
        # map_func = partial(_register_frame, mean_img=mean_img,
        #                 upsample_factor=upsample_factor)
        # results = pool.map(map_func, frames_for_mean)
        # pool.close()
        # pool.join()

        results = [_register_frame(frame, mean_img=mean_img,
                           upsample_factor=upsample_factor) for frame in frames_for_mean]

        # preallocate the results array
        mean_img_dx = np.zeros(num_images_for_mean, dtype=np.float)
        mean_img_dy = np.zeros(num_images_for_mean, dtype=np.float)

        # get results (0: dx, 1: dy, 2: registered image)
        for idx, result in enumerate(results):
            mean_img_dx[idx] = result[0]
            mean_img_dy[idx] = result[1]
            frames_for_mean[idx] = result[2] # overwrite the frames used to make mean image

        # make the new (improved) mean image
        mean_img = np.mean(frames_for_mean, 0)
        mean_img_err = np.mean(
            np.absolute(mean_img_dx)) + np.mean(np.absolute(mean_img_dy))

        if verbose_state:
            print('    Iteration ' + str(iteration) +
                  ', average error: ' + str(mean_img_err) + ' pixels')
        iteration += 1

    return mean_img


def _register_all_frames(input_array, mean_img, upsample_factor=10):
    """
    Register all input frames to the computed aligned mean image.

    Returns
    -------
    dx : float array
        array of x pixel offsets for each frame
    dy : float array
        array of y pixel offsets for each frame
    registered_frames : np.ndarray (size of input images)
        array containing each aligned frame
    """
    if verbose_state:
        print('Registering all ' + str(im_dim[0]) + ' frames...')

    # configure pool of workers (multiprocessing)
    # pool = multiprocessing.Pool(n_workers)
    # map_func = partial(
    #     _register_frame, mean_img=mean_img, upsample_factor=upsample_factor)
    # results = pool.map(map_func, input_array)
    # pool.close()
    # pool.join()

    results = [_register_frame(frame, mean_img=mean_img,
                   upsample_factor=upsample_factor) for frame in input_array]

    # preallocate arrays
    dx = np.zeros(im_dim[0], dtype=np.float)
    dy = np.zeros(im_dim[0], dtype=np.float)
    registered_frames = np.zeros(im_dim, dtype=im_dtype)

    # get results (0: dx, 1: dy, 2: registered image)
    for idx, result in enumerate(results):
        dx[idx] = result[0]
        dy[idx] = result[1]
        registered_frames[idx] = result[2]

    return dx, dy, registered_frames


def _register_frame(frame, mean_img, upsample_factor=10):
    """
    Called by _make_mean_img and _register_all_frames
    """

    # compute the offsets
    shifts = _register_translation(
        mean_img, frame, upsample_factor=upsample_factor)

    # shift the frame
    dy, dx = shifts
    registered_frame = shift(
        frame, [dy, dx], order=3, mode='constant', cval=0, output=im_dtype)  # output forced to np.uint16 - if using detected type (float, im_dtype) weird things happen for presumably black pixel values when registered movie is saved with the built in method

    return dx, dy, registered_frame


def _get_ffts(use_fftw=False, nthreads=1,):
    """
    ********************************
    Modified from image_registration
    ********************************
    Decide which fftn and ifftn methods to use. Either use standard numpy
    or the (slightly) faster fftw (requires PyFFTW3 0.2.1)

    Parameters
    ----------
    use_fftw : bool
        use FFTW if available? (Default = False)
    nthreads : int
        how many threads to use in the returned methods (only relevant
            for fftw)
    Returns
    -------
    fftn,ifftn : methods
        the fft methods to use
    """
    if use_fftw:
        try:
            import fftw3

            def fftn(array, nthreads=1):
                array = array.astype('complex').copy()
                outarray = array.copy()
                fft_forward = fftw3.Plan(array, outarray,
                                         direction='forward',
                                         flags=['estimate'],
                                         nthreads=nthreads)
                fft_forward.execute()
                return outarray

            def ifftn(array, nthreads=1):
                array = array.astype('complex').copy()
                outarray = array.copy()
                fft_backward = fftw3.Plan(array, outarray,
                                          direction='backward',
                                          flags=['estimate'],
                                          nthreads=nthreads)
                fft_backward.execute()
                return outarray / np.size(array)
            if verbose_state:
                print('Using FFTW')
        except:
            if verbose_state:
                print('PyFFTW3 not found!')
            fftn = np.fft.fftn
            ifftn = np.fft.ifftn
    else:
        fftn = np.fft.fftn
        ifftn = np.fft.ifftn

    return fftn, ifftn


def _upsampled_dft(data, upsampled_region_size,
                   upsample_factor=1, axis_offsets=None):
    """
    *****************************************
    From skimage.feature.register_translation
    *****************************************
    Upsampled DFT by matrix multiplication.
    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.
    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.
    Parameters
    ----------
    data : 2D ndarray
        The input data array (DFT of original data) to upsample.
    upsampled_region_size : integer or tuple of integers, optional
        The size of the region to be sampled.  If one integer is provided, it
        is duplicated up to the dimensionality of ``data``.
    upsample_factor : integer, optional
        The upsampling factor.  Defaults to 1.
    axis_offsets : tuple of integers, optional
        The offsets of the region to be sampled.  Defaults to None (uses
        image center)
    Returns
    -------
    output : 2D ndarray
            The upsampled DFT of the specified region.
    """

    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size, ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError("shape of upsampled region sizes must be equal "
                             "to input data's number of dimensions.")

    if axis_offsets is None:
        axis_offsets = [0, ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError("number of axis offsets must be equal to input "
                             "data's number of dimensions.")

    col_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[1] * upsample_factor)) *
        (np.fft.ifftshift(np.arange(data.shape[1]))[:, None] -
         np.floor(data.shape[1] / 2)).dot(
             np.arange(upsampled_region_size[1])[None, :] - axis_offsets[1])
    )
    row_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[0] * upsample_factor)) *
        (np.arange(upsampled_region_size[0])[:, None] - axis_offsets[0]).dot(
            np.fft.ifftshift(np.arange(data.shape[0]))[None, :] -
            np.floor(data.shape[0] / 2))
    )

    return row_kernel.dot(data).dot(col_kernel)  # for some reason this is the point it hangs when multiprocessing


def _compute_phasediff(cross_correlation_max):
    """
    *****************************************
    From skimage.feature.register_translation
    *****************************************
    Compute global phase difference between the two images (should be
        zero if images are non-negative).
    Parameters
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.
    """
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)


def _compute_error(cross_correlation_max, src_amp, target_amp):
    """
    *****************************************
    From skimage.feature.register_translation
    *****************************************
    Compute RMS error metric between ``src_image`` and ``target_image``.
    Parameters
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.
    src_amp : float
        The normalized average image intensity of the source image
    target_amp : float
        The normalized average image intensity of the target image
    """
    error = 1.0 - cross_correlation_max * cross_correlation_max.conj() /\
        (src_amp * target_amp)
    return np.sqrt(np.abs(error))


def _register_translation(src_image, target_image, upsample_factor=1,
                          space="real"):
    """
    *****************************************
    From skimage.feature.register_translation
    *****************************************
    Efficient subpixel image translation registration by cross-correlation.
    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.
    Parameters
    ----------
    src_image : ndarray
        Reference image.
    target_image : ndarray
        Image to register.  Must be same dimensionality as ``src_image``.
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel.  Default is 1 (no upsampling)
    space : string, one of "real" or "fourier"
        Defines how the algorithm interprets input data.  "real" means data
        will be FFT'd to compute the correlation, while "fourier" data will
        bypass FFT of input data.  Case insensitive.
    Returns
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register ``target_image`` with
        ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)
    error : float
        Translation invariant normalized RMS error between ``src_image`` and
        ``target_image``.
    phasediff : float
        Global phase difference between the two images (should be
        zero if images are non-negative).
    References
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008).
    """

    # images must be the same shape
    if src_image.shape != target_image.shape:
        raise ValueError("Error: images must be same size for "
                         "register_translation")

    # only 2D data makes sense right now
    if src_image.ndim != 2 and upsample_factor > 1:
        raise NotImplementedError("Error: register_translation only supports "
                                  "subpixel registration for 2D images")

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_image = np.array(src_image, dtype=np.complex128, copy=False)
        target_image = np.array(target_image, dtype=np.complex128, copy=False)
        src_freq = fftn(src_image)
        target_freq = fftn(target_image)
    else:
        raise ValueError("Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = ifftn(image_product)

    # Locate maximum
    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:
        src_amp = np.sum(np.abs(src_freq) ** 2) / src_freq.size
        target_amp = np.sum(np.abs(target_freq) ** 2) / target_freq.size
        CCmax = cross_correlation.max()
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:

        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor

        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()

        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
            np.argmax(np.abs(cross_correlation)),
            cross_correlation.shape),
            dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + maxima / upsample_factor
        CCmax = cross_correlation.max()
        src_amp = _upsampled_dft(src_freq * src_freq.conj(),
                                 1, upsample_factor)[0, 0]
        src_amp /= normalization
        target_amp = _upsampled_dft(target_freq * target_freq.conj(),
                                    1, upsample_factor)[0, 0]
        target_amp /= normalization

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts
    # _compute_error(CCmax, src_amp, target_amp)
    # _compute_phasediff(CCmax)


def _save_registered_frames(input_array, save_name, save_fmt):
    """
    Save
    Parameters
    ----------
    Returns
    -------
    """
    if verbose_state:
        print('Saving...')
    try:  # this is ugly
        import tifffile
    except ImportError:
        try:
            from sima.misc import tifffile
        except ImportError:
            if verbose_state:
                print('Cannot find tifffile')

    if save_fmt == 'singles':
        for idx in range(im_dim[0]):
            tifffile.imsave(
                save_name + '_' + '{number:05d}'.format(number=idx)
                + '_DFTreg.tif', input_array[idx].astype(np.uint16))
    if save_fmt == 'mptiff':
        tifffile.imsave(save_name + '_DFTreg.tif', input_array.astype(np.uint16))
    elif save_fmt == 'bigtiff':
        tifffile.imsave(save_name + '_DFTreg.tif', input_array.astype(np.uint16), bigtiff=True)


