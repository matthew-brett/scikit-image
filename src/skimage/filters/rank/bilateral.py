"""Approximate bilateral rank filter for local (custom kernel) mean.

The local histogram is computed using a sliding window similar to the method
described in [1]_.

The pixel neighborhood is defined by:

* the given footprint (structuring element)
* an interval [g-s0, g+s1] in graylevel around g the processed pixel graylevel

The kernel is flat (i.e. each pixel belonging to the neighborhood contributes
equally).

Result image is 8-/16-bit or double with respect to the input image and the
rank filter operation.

References
----------

.. [1] Huang, T. ,Yang, G. ;  Tang, G.. "A fast two-dimensional
       median filtering algorithm", IEEE Transactions on Acoustics, Speech and
       Signal Processing, Feb 1979. Volume: 27 , Issue: 1, Page(s): 13 - 18.

"""

from functools import wraps
import inspect
import warnings

from ..._shared.utils import check_nD, _warning_stacklevel
from . import bilateral_cy
from .generic import _preprocess_input

__all__ = ['mean_bilateral', 'pop_bilateral', 'sum_bilateral']


def _apply(func, image, footprint, out, mask, shift_x, shift_y, s0, s1, out_dtype=None):
    check_nD(image, 2)
    image, footprint, out, mask, n_bins = _preprocess_input(
        image,
        footprint,
        out,
        mask,
        out_dtype,
        shift_x=shift_x,
        shift_y=shift_y,
    )

    func(
        image,
        footprint,
        shift_x=shift_x,
        shift_y=shift_y,
        mask=mask,
        out=out,
        n_bins=n_bins,
        s0=s0,
        s1=s1,
    )

    return out.reshape(out.shape[:2])


# Deprecation start and stop for shift_{x,y} and positional after `footprint`.
_SHIFT_X_DEP_START = '1.0'
_SHIFT_X_DEP_STOP = '2.2'


def _proc_shifts(kwargs, stacklevel):
    bad_kwds = {k: kwargs.pop(k, None) for k in ('shift_x', 'shift_y')}
    if not any(v is not None for v in bad_kwds.values()):
        return
    if 'shift' in kwargs:
        raise ValueError('Cannot mix `shift` and either `shift_x` or `shift_y`')
    warnings.warn(
        '`shift_x` and `shift_y` are deprecated since version '
        f'{_SHIFT_X_DEP_START}, and will be removed in version '
        f'{_SHIFT_X_DEP_STOP}; use `shift` instead, where the new argument'
        'can be constructed as `shift=(shift_y, shift_x)`. '
        '*Note*: the `shift_y` value is first in the tuple.',
        FutureWarning,
        stacklevel=stacklevel,
    )
    kwargs['shift'] = [bad_kwds[n] or 0 for n in ('shift_y', 'shift_x')]


SK2_TO_1_ATTR = '__sk2_to_1__'


def label_2to1(dec):
    """Label wrapping decorator as being 2 to 1 converter."""

    def labeled_dec(func):
        out_func = dec(func)
        setattr(out_func, SK2_TO_1_ATTR, True)
        return out_func

    return labeled_dec


def is_2to1(func):
    """True if this function is a 2 to 1 wrapper."""
    return getattr(func, SK2_TO_1_ATTR, False)


def unwrap_2to1(func):
    if hasattr(func, '__wrapped__') and is_2to1(func):
        return func.__wrapped__
    return func


@label_2to1
def shift_xy_to_shift(in_func):
    out_params = list(inspect.signature(in_func).parameters)
    assert out_params[4] == 'shift'
    stacklevel = _warning_stacklevel(in_func)

    @wraps(in_func)
    def out_func(*args, **kwargs):
        if len(args) > 2:
            warnings.warn(
                f'All *positional* arguments to `{in_func.__name__}` after '
                '`footprint` are deprecated since version '
                f'{_SHIFT_X_DEP_START} and will be keyword-only from version '
                f'{_SHIFT_X_DEP_STOP}.',
                FutureWarning,
                stacklevel=stacklevel,
            )
        for i, arg in enumerate(args):
            kwargs[out_params[i]] = arg
        _proc_shifts(kwargs, stacklevel)
        return in_func(**kwargs)

    return out_func


@shift_xy_to_shift
def mean_bilateral(
    image,
    footprint,
    out=None,
    mask=None,
    *,
    shift=(0, 0),
    s0=10,
    s1=10,
    # image, footprint, out=None, mask=None, shift_x=0, shift_y=0, s0=10, s1=10
):
    """Apply a flat kernel bilateral filter.

    This is an edge-preserving and noise reducing denoising filter. It averages
    pixels based on their spatial closeness and radiometric similarity.

    Spatial closeness is measured by considering only the local pixel
    neighborhood given by a footprint (structuring element).

    Radiometric similarity is defined by the graylevel interval [g-s0, g+s1]
    where g is the current pixel graylevel.

    Only pixels belonging to the footprint and having a graylevel inside this
    interval are averaged.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    footprint : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array, same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift : sequence of int
        Offset added to the `footprint` center point, where first value in
        sequence is offset on first axis of `footprint` array, and second is
        offset on second. Shift is bounded to the footprint sizes (center must
        be inside the given footprint).
    s0, s1 : int
        Define the [s0, s1] interval around the grayvalue of the center pixel
        to be considered for computing the value.

    Returns
    -------
    out : 2-D array, same dtype as input `image`
        Output image.

    See also
    --------
    skimage.restoration.denoise_bilateral

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filters.rank import mean_bilateral
    >>> img = data.camera().astype(np.uint16)
    >>> bilat_img = mean_bilateral(img, disk(20), s0=10,s1=10)

    """
    print('Shift', shift)
    return _apply(
        bilateral_cy._mean,
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift[1],
        shift_y=shift[0],
        s0=s0,
        s1=s1,
    )


def pop_bilateral(
    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, s0=10, s1=10
):
    """Return the local number (population) of pixels.


    The number of pixels is defined as the number of pixels which are included
    in the footprint and the mask. Additionally pixels must have a graylevel
    inside the interval [g-s0, g+s1] where g is the grayvalue of the center
    pixel.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    footprint : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array, same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    s0, s1 : int
        Define the [s0, s1] interval around the grayvalue of the center pixel
        to be considered for computing the value.

    Returns
    -------
    out : 2-D array, same dtype as input `image`
        Output image.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import footprint_rectangle
    >>> import skimage.filters.rank as rank
    >>> img = 255 * np.array([[0, 0, 0, 0, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> rank.pop_bilateral(img, footprint_rectangle((3, 3)), s0=10, s1=10)
    array([[3, 4, 3, 4, 3],
           [4, 4, 6, 4, 4],
           [3, 6, 9, 6, 3],
           [4, 4, 6, 4, 4],
           [3, 4, 3, 4, 3]], dtype=uint16)

    """

    return _apply(
        bilateral_cy._pop,
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        s0=s0,
        s1=s1,
    )


def sum_bilateral(
    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, s0=10, s1=10
):
    """Apply a flat kernel bilateral filter.

    This is an edge-preserving and noise reducing denoising filter. It averages
    pixels based on their spatial closeness and radiometric similarity.

    Spatial closeness is measured by considering only the local pixel
    neighborhood given by a footprint (structuring element).

    Radiometric similarity is defined by the graylevel interval [g-s0, g+s1]
    where g is the current pixel graylevel.

    Only pixels belonging to the footprint AND having a graylevel inside this
    interval are summed.

    Note that the sum may overflow depending on the data type of the input
    array.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    footprint : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array, same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    s0, s1 : int
        Define the [s0, s1] interval around the grayvalue of the center pixel
        to be considered for computing the value.

    Returns
    -------
    out : 2-D array, same dtype as input `image`
        Output image.

    See also
    --------
    skimage.restoration.denoise_bilateral

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filters.rank import sum_bilateral
    >>> img = data.camera().astype(np.uint16)
    >>> bilat_img = sum_bilateral(img, disk(10), s0=10, s1=10)

    """

    return _apply(
        bilateral_cy._sum,
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        s0=s0,
        s1=s1,
    )
