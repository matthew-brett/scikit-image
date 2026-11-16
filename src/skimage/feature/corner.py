from _skimage2._shared.utils import warn
from _skimage2.feature.corner import (
    _symmetric_image,
    corner_fast as corner_fast,
    corner_foerstner as corner_foerstner,
    corner_harris as corner_harris,
    corner_kitchen_rosenfeld as corner_kitchen_rosenfeld,
    corner_moravec as corner_moravec,
    corner_orientations as corner_orientations,
    corner_shi_tomasi as corner_shi_tomasi,
    corner_subpix as corner_subpix,
    hessian_matrix as _ski2_hessian_matrix,
    hessian_matrix_det as _ski2_hessian_matrix_det,
    hessian_matrix_eigvals as hessian_matrix_eigvals,
    shape_index as _ski2_shape_index,
    structure_tensor as structure_tensor,
    structure_tensor_eigenvalues as structure_tensor_eigenvalues,
)  # noqa: F401

__all__ = [
    'corner_fast',
    'corner_foerstner',
    'corner_harris',
    'corner_kitchen_rosenfeld',
    'corner_moravec',
    'corner_orientations',
    'corner_peaks',
    'corner_shi_tomasi',
    'corner_subpix',
    'hessian_matrix',
    'hessian_matrix_det',
    'hessian_matrix_eigvals',
    'shape_index',
    'structure_tensor',
    'structure_tensor_eigenvalues',
]

import numpy as np

from _skimage2.feature.corner import corner_peaks as ski2_corner_peaks

from skimage._migration import ski2_migration_decorator


@ski2_migration_decorator(
    """\
``%(qname_old)s`` will be removed in scikit-image 2. Please use
``skimage2.feature.peak_local_max`` instead.
""",
    qname_old='skimage.feature.corner_peaks',
)
def corner_peaks(
    image,
    min_distance=1,
    threshold_abs=None,
    threshold_rel=None,
    exclude_border=True,
    indices=True,
    num_peaks=np.inf,
    footprint=None,
    labels=None,
    *,
    num_peaks_per_label=np.inf,
    p_norm=np.inf,
):
    """Find peaks in corner measure response image.

    This differs from `skimage.feature.peak_local_max` in that it suppresses
    multiple connected peaks with the same accumulator value.

    Parameters
    ----------
    image : ndarray of shape (M, N)
        Input image.
    min_distance : int, optional
        The minimal allowed distance separating peaks.
    * : *
        See :py:meth:`skimage.feature.peak_local_max`.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.

    Returns
    -------
    output : ndarray or ndarray of bools

        * If `indices = True`  : (row, column, ...) coordinates of peaks.
        * If `indices = False` : Boolean array shaped like `image`, with peaks
          represented by True values.

    See also
    --------
    skimage.feature.peak_local_max

    Notes
    -----
    .. versionchanged:: 0.18
        The default value of `threshold_rel` has changed to None, which
        corresponds to letting `skimage.feature.peak_local_max` decide on the
        default. This is equivalent to `threshold_rel=0`.

    The `num_peaks` limit is applied before suppression of connected peaks.
    To limit the number of peaks after suppression, set `num_peaks=np.inf` and
    post-process the output of this function.

    Examples
    --------
    >>> from skimage.feature import peak_local_max
    >>> response = np.zeros((5, 5))
    >>> response[2:4, 2:4] = 1
    >>> response
    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 1., 1., 0.],
           [0., 0., 1., 1., 0.],
           [0., 0., 0., 0., 0.]])
    >>> peak_local_max(response)
    array([[2, 2],
           [2, 3],
           [3, 2],
           [3, 3]])
    >>> corner_peaks(response)
    array([[2, 2]])

    """
    # Allow exclude_border=True|False
    exclude_border = (
        0
        if exclude_border is False
        else (int(np.floor(min_distance)) if exclude_border is True else exclude_border)
    )
    return ski2_corner_peaks(
        image,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        exclude_border=exclude_border,
        indices=indices,
        num_peaks=num_peaks,
        footprint=footprint,
        labels=labels,
        num_peaks_per_label=num_peaks_per_label,
        p_norm=p_norm,
    )


_HESSIAN_MATRIX_MIGRATION = """\
<!--- cond-start: doc -->
``%(qname_new)s`` takes ``method`` instead of ``use_gaussian_derivatives``,
and it has no ``order`` parameter.

``method='gaussian-derivative'`` is the default. It replaces
``use_gaussian_derivatives=True``; those calls drop the keyword.

``method='finite-difference'`` replaces ``use_gaussian_derivatives=False``,
including a call that left the argument at its default.

``%(qname_new)s`` returns the ``order='rc'`` element list. ``order='xy'``
is that list reversed, for either method:

>>> import numpy as np
>>> import skimage as ski1
>>> import skimage2 as ski2
>>> image = ski1.data.camera()[::4, ::4]
>>> xy = ski1.feature.hessian_matrix(
...     image, sigma=2, order='xy', use_gaussian_derivatives=True)
>>> rc = ski2.feature.hessian_matrix(image, sigma=2)
>>> all(np.array_equal(a, b) for a, b in zip(xy, rc[::-1]))
True
<!--- cond-end -->
"""


_SHAPE_INDEX_MIGRATION = """\
<!--- cond-start: doc -->
``%(qname_new)s`` still builds its Hessian with ``method='finite-difference'``,
so the values match. The Gaussian-derivative curvatures are
``skimage2.feature.hessian_matrix(..., method='gaussian-derivative')``.
<!--- cond-end -->
"""


_HESSIAN_MATRIX_DET_MIGRATION = """\
<!--- cond-start: doc -->
``approximate=True`` on a 2-D image is unchanged: the box-filter approximation.

``approximate=False`` in ``%(qname_new)s`` is the determinant of
``skimage2.feature.hessian_matrix``, whose default method is
``'gaussian-derivative'``. ``%(qname_old)s`` uses finite differences there.

``approximate=True`` on a 3-D image raises ``ValueError`` in
``%(qname_new)s``. ``%(qname_old)s`` ignores ``approximate`` for a volume and
returns the finite-difference determinant. Those numbers are the determinant
of ``skimage2.feature.hessian_matrix(..., method='finite-difference')``.
``approximate=False`` is the Gaussian-derivative determinant in every rank.
<!--- cond-end -->
"""


@ski2_migration_decorator(
    _HESSIAN_MATRIX_MIGRATION, qname_old='skimage.feature.hessian_matrix'
)
def hessian_matrix(
    image, sigma=1, mode='constant', cval=0, order='rc', use_gaussian_derivatives=None
):
    """Compute the Hessian matrix.

    In 2D, the Hessian matrix is defined as::

        H = [Hrr Hrc]
            [Hrc Hcc]

    which is computed by convolving the image with the second derivatives
    of the Gaussian kernel in the respective r- and c-directions.

    The implementation here also supports n-dimensional data.

    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    order : {'rc', 'xy'}, optional
        For 2D images, this parameter allows for the use of reverse or forward
        order of the image axes in gradient computation. 'rc' indicates the use
        of the first axis initially (Hrr, Hrc, Hcc), whilst 'xy' indicates the
        usage of the last axis initially (Hxx, Hxy, Hyy). Images with higher
        dimension must always use 'rc' order.
    use_gaussian_derivatives : bool, optional
        Indicates whether the Hessian is computed by convolving with Gaussian
        derivatives, or by a simple finite-difference operation.

    Returns
    -------
    H_elems : list of ndarray
        Upper-diagonal elements of the hessian matrix for each pixel in the
        input image. In 2D, this will be a three element list containing [Hrr,
        Hrc, Hcc]. In nD, the list will contain ``(n**2 + n) / 2`` arrays.

    Notes
    -----
    When ``use_gaussian_derivatives`` is ``True``, each element is one separable
    convolution with a moment-corrected Gaussian derivative, so the border
    rule is applied once. ``order='xy'`` is that element list reversed.

    When ``use_gaussian_derivatives`` is ``False``, simple finite differences
    on a Gaussian-smoothed image are used instead.

    .. versionchanged:: 0.27
        With ``use_gaussian_derivatives=True``, each element is computed in one
        pass. The previous scheme differentiated twice.

    Examples
    --------
    >>> from skimage.feature import hessian_matrix
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> Hrr, Hrc, Hcc = hessian_matrix(square, sigma=0.1, order='rc',
    ...                                use_gaussian_derivatives=False)
    >>> Hrc
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0., -1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0., -1.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    """
    if np.asarray(image).ndim > 2 and order == 'xy':
        raise ValueError("order='xy' is only supported for 2D images.")
    if order not in ('rc', 'xy'):
        raise ValueError(f'unrecognized order: {order}')
    if use_gaussian_derivatives is None:
        use_gaussian_derivatives = False
        warn(
            'use_gaussian_derivatives currently defaults to False, but will '
            'change to True in a future version. Please specify this '
            'argument explicitly to maintain the current behavior',
            category=FutureWarning,
            stacklevel=2,
        )
    method = 'gaussian-derivative' if use_gaussian_derivatives else 'finite-difference'
    elements = _ski2_hessian_matrix(
        image, sigma=sigma, mode=mode, cval=cval, method=method
    )
    return elements[::-1] if order == 'xy' else elements


@ski2_migration_decorator(
    _SHAPE_INDEX_MIGRATION, qname_old='skimage.feature.shape_index'
)
def shape_index(image, sigma=1, mode='constant', cval=0):
    """Compute the shape index.

    See ``_skimage2.feature.shape_index`` for the definition. This function
    returns the same values.
    """
    return _ski2_shape_index(image, sigma=sigma, mode=mode, cval=cval)


shape_index.__doc__ = _ski2_shape_index.__doc__.replace(
    'from _skimage2.feature import shape_index',
    'from skimage.feature import shape_index',
)


@ski2_migration_decorator(
    _HESSIAN_MATRIX_DET_MIGRATION, qname_old='skimage.feature.hessian_matrix_det'
)
def hessian_matrix_det(image, sigma=1, approximate=True):
    """Compute the approximate Hessian Determinant over an image.

    The 2D approximate method uses box filters over integral images to
    compute the approximate Hessian Determinant.

    Parameters
    ----------
    image : ndarray
        The image over which to compute the Hessian Determinant.
    sigma : float, optional
        Standard deviation of the Gaussian kernel used for the Hessian
        matrix.
    approximate : bool, optional
        If ``True`` and the image is 2D, use a much faster approximate
        computation. This argument has no effect on 3D and higher images.

    Returns
    -------
    out : array
        The array of the Determinant of Hessians.

    Notes
    -----
    For 2D images when ``approximate=True``, the running time of this method
    only depends on size of the image. It is independent of `sigma` as one
    would expect. The downside is that the result for `sigma` less than `3`
    is not accurate, i.e., not similar to the result obtained if someone
    computed the Hessian and took its determinant.

    Otherwise the determinant is that of the finite-difference Hessian.
    """
    if np.asarray(image).ndim == 2 and approximate:
        return _ski2_hessian_matrix_det(image, sigma=sigma, approximate=True)
    elements = _ski2_hessian_matrix(image, sigma=sigma, method='finite-difference')
    return np.linalg.det(_symmetric_image(elements))


# The re-exported eigenvalue routine's example calls ``_skimage2``'s
# ``hessian_matrix``. In this module that name is the v1 wrapper.
__doctest_skip__ = [
    '_ski2_hessian_matrix',
    '_ski2_hessian_matrix_det',
    '_ski2_shape_index',
    '_symmetric_image',
    'hessian_matrix_eigvals',
]


from skimage._doctest_adapters import adapt_doctests  # noqa: E402

adapt_doctests(globals())
