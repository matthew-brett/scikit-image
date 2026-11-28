"""The SURF box-filter determinant of ``hessian_matrix_det(approximate=True)``.

The filters of SURF [1] for a lobe ``L`` (an odd integer) are ``3 L`` pixels
wide. ``Dxx`` weighs three lobes of ``L`` columns by 1, −2 and 1, over
``2 L − 1`` rows; ``Dyy`` is its transpose; ``Dxy`` weighs four ``L × L``
squares, one pixel off the centre row and column, by +1 and −1. Each is
normalised by ``1 / (3 L)²``, and the determinant is
``Dxx Dyy − 0.81 Dxy²``.

The tests check three things:

1. The implementation against a direct construction of those kernels,
   correlated with the image with zeros outside it (`_reference_det`).
2. Properties that the filters ought to have: zero response to a constant and
   to a ramp, invariance to the sign of the image, equivariance under
   transposition, and symmetry about the centre of a symmetric image.
3. Agreement with the scale-normalised determinant ``s⁴ det H`` of an exact
   Gaussian Hessian, on Gaussian sums fitted to photographs
   (``gaussian_sum_images``, ``tests/gaussian_sum.py``), at ``s = 0.55 L``.
   The box filters are an approximation, so the test bounds the error after
   the best common gain, and checks that no one-pixel shift of the output
   fits better.

References
----------
.. [1] H. Bay, A. Ess, T. Tuytelaars and L. Van Gool, "Speeded-Up Robust
    Features (SURF)," Comput. Vis. Image Underst., 110(3):346–359, 2008,
    §3.1. https://doi.org/10.1016/j.cviu.2007.09.014
"""

import numpy as np
import pytest
from scipy import ndimage as ndi

from skimage.draw import disk
from skimage.feature import blob_doh, hessian_matrix_det
from skimage.feature._hessian_det_appx import _surf_lobe

from ... import gaussian_sum

GAUSSIAN_PER_LOBE = 0.55  # Gaussian standard deviation that lobe 1 matches


def _kernels(lobe):
    """``(Dxx, Dyy, Dxy)`` kernels for `lobe`, centred in a ``3 L`` square."""
    size = 3 * lobe
    mid = size // 2
    rows = slice(mid - lobe + 1, mid + lobe)  # 2 L - 1 rows about the centre
    centre_lobe = slice(mid - lobe // 2, mid + lobe // 2 + 1)
    dxx = np.zeros((size, size))
    dxx[rows, :] = 1
    dxx[rows, centre_lobe] = -2
    before, after = slice(mid - lobe, mid), slice(mid + 1, mid + lobe + 1)
    dxy = np.zeros((size, size))
    dxy[before, after] = dxy[after, before] = 1
    dxy[before, before] = dxy[after, after] = -1
    return dxx, dxx.T, dxy


def _reference_det(image, sigma):
    """SURF determinant by correlation with the kernels, zero outside."""
    lobe = _surf_lobe(sigma)
    dxx, dyy, dxy = (
        ndi.correlate(image, k, mode='constant') / (3 * lobe) ** 2
        for k in _kernels(lobe)
    )
    return dxx * dyy - 0.81 * dxy**2


SIGMAS = [0.5, 1, 2, 2.5, 3, 3.7, 4.2, 5.6, 7, 9.9]


@pytest.mark.parametrize(
    ('sigma', 'lobe'),
    [(0.3, 1), (1, 1), (1.9, 1), (2, 3), (3, 3), (3.9, 3), (4, 5), (4.2, 5), (7, 7)],
)
def test_lobe_is_nearest_odd_integer(sigma, lobe):
    assert _surf_lobe(sigma) == lobe


@pytest.mark.parametrize('sigma', SIGMAS)
def test_matches_reference_at_every_pixel(sigma):
    """Same as the kernel construction, border included, for signed images."""
    image = np.random.default_rng(0).normal(size=(40, 48))
    expected = _reference_det(image, sigma)
    got = hessian_matrix_det(image, sigma=sigma)
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-12 * np.abs(expected).max()
    )


@pytest.mark.parametrize('sigma', SIGMAS)
def test_zero_for_constant_and_ramp(sigma):
    """Each filter sums to zero and is centred, so a ramp gives zero inside."""
    reach = 3 * _surf_lobe(sigma)
    inner = (slice(reach, -reach),) * 2
    r, c = np.mgrid[:64, :64].astype(float)
    for image in (np.full((64, 64), 0.7), 0.01 * r + 0.02 * c):
        assert np.abs(hessian_matrix_det(image, sigma=sigma)[inner]).max() < 1e-12


@pytest.mark.parametrize('sigma', SIGMAS)
def test_sign_and_transpose(sigma):
    image = np.random.default_rng(1).normal(size=(40, 48))
    det = hessian_matrix_det(image, sigma=sigma)
    np.testing.assert_allclose(hessian_matrix_det(-image, sigma=sigma), det, atol=1e-12)
    np.testing.assert_allclose(
        hessian_matrix_det(image.T, sigma=sigma), det.T, atol=1e-12
    )


@pytest.mark.parametrize('sigma', SIGMAS)
def test_border_counts_zero_outside(sigma):
    """The output equals that of the image padded with zeros, then cropped."""
    image = np.random.default_rng(2).uniform(size=(30, 35))
    pad = 3 * _surf_lobe(sigma)
    padded = np.pad(image, pad)
    expected = hessian_matrix_det(padded, sigma=sigma)[pad:-pad, pad:-pad]
    np.testing.assert_allclose(
        hessian_matrix_det(image, sigma=sigma), expected, atol=1e-12
    )


@pytest.mark.parametrize('sigma', [3, 4.2, 9, 17])
def test_symmetric_about_disk_centre(sigma):
    """A disk symmetric about a pixel gives a response symmetric about it."""
    image = np.zeros((101, 101))
    image[disk((50, 50), 16)] = 1
    det = hessian_matrix_det(image, sigma=sigma)
    np.testing.assert_allclose(det, det[::-1, ::-1], atol=1e-12)
    np.testing.assert_allclose(det, det[::-1], atol=1e-12)


def _shape_error(got, truth):
    """Relative RMS error of `got` after the best common gain onto `truth`."""
    gain = np.sum(got * truth) / np.sum(truth * truth)
    return np.sqrt(np.sum((got - gain * truth) ** 2) / np.sum(truth * truth))


def _gold_comparison(levels, image, lobe):
    """Box determinant and exact ``s⁴ det H`` at ``s = 0.55 L``, inside the frame."""
    s = GAUSSIAN_PER_LOBE * lobe
    Hrr, Hrc, Hcc = gaussian_sum.hessian(levels, image.shape, s)
    exact = s**4 * (Hrr * Hcc - Hrc**2)
    box = hessian_matrix_det(image, sigma=lobe)
    reach = int(max(8 * s, 1.5 * lobe)) + 2
    inner = (slice(reach, -reach),) * 2
    return box, exact, inner


@pytest.mark.parametrize('name', ['camera', 'coins', 'astronaut'])
@pytest.mark.parametrize(
    ('lobe', 'floor'),
    [(3, 0.15), (5, 0.25), (7, 0.25), (9, 0.25), (13, 0.25)],
)
def test_matches_gold_standard_shape(gaussian_sum_images, name, lobe, floor):
    """Shape error against ``s⁴ det H`` at ``s = 0.55 L``, after the best gain.

    Measured: 0.08 to 0.11 at lobe 3, 0.15 to 0.23 at lobes 5 to 13.
    """
    levels, image = gaussian_sum_images[name]
    box, exact, inner = _gold_comparison(levels, image, lobe)
    assert _shape_error(box[inner], exact[inner]) < floor


@pytest.mark.parametrize('name', ['camera', 'coins', 'astronaut'])
@pytest.mark.parametrize('lobe', [3, 5, 9, 13])
def test_centred_on_gold_standard(gaussian_sum_images, name, lobe):
    """No one-pixel shift of the output fits the exact determinant better."""
    levels, image = gaussian_sum_images[name]
    box, exact, inner = _gold_comparison(levels, image, lobe)
    unshifted = _shape_error(box[inner], exact[inner])
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr, dc) != (0, 0):
                shifted = np.roll(box, (dr, dc), axis=(0, 1))
                assert unshifted < _shape_error(shifted[inner], exact[inner])


def test_blob_doh_one_scale_per_lobe():
    """Each lobe is one scale, so sigma is the odd lobe just above the radius.

    With a plane for each of 30 sigma, identical neighbouring planes would
    give false maxima across scale.
    """
    discs = [(80, 80, 6), (80, 220, 10), (220, 80, 16), (220, 260, 24)]
    image = np.zeros((400, 400))
    for r, c, radius in discs:
        image[disk((r, c), radius)] = 1
    blobs = blob_doh(image, min_sigma=2, max_sigma=30, num_sigma=30, threshold=0.005)
    assert len(blobs) == len(discs)
    for r, c, radius in discs:
        blob = blobs[np.argmin(np.hypot(blobs[:, 0] - r, blobs[:, 1] - c))]
        # The response is flat over a pixel or two at the centre, and
        # peak_local_max picks one pixel of the plateau.
        assert np.hypot(blob[0] - r, blob[1] - c) <= 1
        assert blob[2] == radius + 1
