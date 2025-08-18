# Shift-x, shift-y

## `_percentile.py`

- `autolevel_percentile`
- `gradient_percentile`
- `mean_percentile`
- `subtract_mean_percentile`
- `enhance_contrast_percentile`
- `percentile`
- `pop_percentile`
- `sum_percentile`
- `threshold_percentile`

## `_generic.py`

- `autolevel`
- `equalize`
- `gradient`
- `maximum`
- `mean`
- `geometric_mean`
- `subtract_mean`
- `median`
- `minimum`
- `modal`
- `enhance_contrast`
- `pop`
- `threshold`
- `noise_filter`
- `entropy`
- `otsu`
- `majority`
- `sum`
- `windowed_histogram`

## `bilateral.py`

- `mean_bilateral`
- `sum_bilateral`
- `pop_bilateral`

## Log

```
src/skimage/filters/rank/core_cy.pxd:                signed char shift_x, signed char shift_y,
src/skimage/filters/rank/bilateral_cy.pyx:          signed char shift_x, signed char shift_y, Py_ssize_t s0, Py_ssize_t s1,
src/skimage/filters/rank/bilateral_cy.pyx:          shift_x, shift_y, 0, 0, s0, s1, n_bins)
src/skimage/filters/rank/bilateral_cy.pyx:         signed char shift_x, signed char shift_y, Py_ssize_t s0, Py_ssize_t s1,
src/skimage/filters/rank/bilateral_cy.pyx:          shift_x, shift_y, 0, 0, s0, s1, n_bins)
src/skimage/filters/rank/bilateral_cy.pyx:         signed char shift_x, signed char shift_y, Py_ssize_t s0, Py_ssize_t s1,
src/skimage/filters/rank/bilateral_cy.pyx:          shift_x, shift_y, 0, 0, s0, s1, n_bins)
src/skimage/filters/rank/_percentile.py:def _apply(func, image, footprint, out, mask, shift_x, shift_y, p0, p1, out_dtype=None):
src/skimage/filters/rank/_percentile.py:        shift_x=shift_x,
src/skimage/filters/rank/_percentile.py:        shift_x=shift_x,
src/skimage/filters/rank/_percentile.py:    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1
src/skimage/filters/rank/_percentile.py:    shift_x, shift_y : int
src/skimage/filters/rank/_percentile.py:        shift_x=shift_x,
src/skimage/filters/rank/_percentile.py:    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1
src/skimage/filters/rank/_percentile.py:    shift_x, shift_y : int
src/skimage/filters/rank/_percentile.py:        shift_x=shift_x,
src/skimage/filters/rank/_percentile.py:    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1
src/skimage/filters/rank/_percentile.py:    shift_x, shift_y : int
src/skimage/filters/rank/_percentile.py:        shift_x=shift_x,
src/skimage/filters/rank/_percentile.py:    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1
src/skimage/filters/rank/_percentile.py:    shift_x, shift_y : int
src/skimage/filters/rank/_percentile.py:        shift_x=shift_x,
src/skimage/filters/rank/_percentile.py:    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1
src/skimage/filters/rank/_percentile.py:    shift_x, shift_y : int
src/skimage/filters/rank/_percentile.py:        shift_x=shift_x,
src/skimage/filters/rank/_percentile.py:def percentile(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0):
src/skimage/filters/rank/_percentile.py:    shift_x, shift_y : int
src/skimage/filters/rank/_percentile.py:        shift_x=shift_x,
src/skimage/filters/rank/_percentile.py:    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1
src/skimage/filters/rank/_percentile.py:    shift_x, shift_y : int
src/skimage/filters/rank/_percentile.py:        shift_x=shift_x,
src/skimage/filters/rank/_percentile.py:    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1
src/skimage/filters/rank/_percentile.py:    shift_x, shift_y : int
src/skimage/filters/rank/_percentile.py:        shift_x=shift_x,
src/skimage/filters/rank/_percentile.py:    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0
src/skimage/filters/rank/_percentile.py:    shift_x, shift_y : int
src/skimage/filters/rank/_percentile.py:        shift_x=shift_x,
tests/skimage/registration/test_masked_phase_cross_correlation.py:        shift_y, shift_x = masked_register_translation(
tests/skimage/registration/test_masked_phase_cross_correlation.py:        assert_equal((shift_x, shift_y), (-xi, yi))
src/skimage/filters/rank/generic.py:    shift_x=None,
src/skimage/filters/rank/generic.py:    shift_x, shift_y : int, optional
src/skimage/filters/rank/generic.py:    for name, value in zip(("shift_x", "shift_y"), (shift_x, shift_y)):
src/skimage/filters/rank/generic.py:    shift_x=None,
src/skimage/filters/rank/generic.py:    shift_x, shift_y, shift_z : int, optional
src/skimage/filters/rank/generic.py:        ("shift_x", "shift_y", "shift_z"), (shift_x, shift_y, shift_z)
src/skimage/filters/rank/generic.py:    func, image, footprint, out, mask, shift_x, shift_y, out_dtype=None
src/skimage/filters/rank/generic.py:    shift_x, shift_y : int
src/skimage/filters/rank/generic.py:        image, footprint, out, mask, out_dtype, shift_x=shift_x, shift_y=shift_y
src/skimage/filters/rank/generic.py:        shift_x=shift_x,
src/skimage/filters/rank/generic.py:    func, image, footprint, out, mask, shift_x, shift_y, shift_z, out_dtype=None
src/skimage/filters/rank/generic.py:        shift_x=shift_x,
src/skimage/filters/rank/generic.py:        shift_x=shift_x,
src/skimage/filters/rank/generic.py:    func, image, footprint, out, mask, shift_x, shift_y, out_dtype=None, pixel_size=1
src/skimage/filters/rank/generic.py:    shift_x, shift_y : int
src/skimage/filters/rank/generic.py:        shift_x=shift_x,
src/skimage/filters/rank/generic.py:        shift_x=shift_x,
src/skimage/filters/rank/generic.py:def autolevel(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
src/skimage/filters/rank/generic.py:    shift_x, shift_y, shift_z : int
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:def equalize(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
src/skimage/filters/rank/generic.py:    shift_x, shift_y, shift_z : int
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:def gradient(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
src/skimage/filters/rank/generic.py:    shift_x, shift_y, shift_z : int
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:def maximum(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
src/skimage/filters/rank/generic.py:    shift_x, shift_y, shift_z : int
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:def mean(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
src/skimage/filters/rank/generic.py:    shift_x, shift_y, shift_z : int
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0
src/skimage/filters/rank/generic.py:    shift_x, shift_y, shift_z : int
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0
src/skimage/filters/rank/generic.py:    shift_x, shift_y, shift_z : int
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:    shift_x=0,
src/skimage/filters/rank/generic.py:    shift_x, shift_y, shift_z : int
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:def minimum(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
src/skimage/filters/rank/generic.py:    shift_x, shift_y, shift_z : int
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:def modal(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
src/skimage/filters/rank/generic.py:    shift_x, shift_y, shift_z : int
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0
src/skimage/filters/rank/generic.py:    shift_x, shift_y, shift_z : int
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:def pop(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
src/skimage/filters/rank/generic.py:    shift_x, shift_y, shift_z : int
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:def sum(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
src/skimage/filters/rank/generic.py:    shift_x, shift_y, shift_z : int
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:def threshold(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
src/skimage/filters/rank/generic.py:    shift_x, shift_y, shift_z : int
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0
src/skimage/filters/rank/generic.py:    shift_x, shift_y, shift_z : int
src/skimage/filters/rank/generic.py:        centre_c = int(footprint.shape[1] / 2) + shift_x
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:        centre_c = int(footprint.shape[1] / 2) + shift_x
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:def entropy(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
src/skimage/filters/rank/generic.py:    shift_x, shift_y, shift_z : int
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:def otsu(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
src/skimage/filters/rank/generic.py:    shift_x, shift_y, shift_z : int
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, n_bins=None
src/skimage/filters/rank/generic.py:    shift_x, shift_y : int, optional
src/skimage/filters/rank/generic.py:        shift_x=shift_x,
src/skimage/filters/rank/generic.py:    shift_x=0,
src/skimage/filters/rank/generic.py:    shift_x, shift_y : int, optional
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/generic.py:            shift_x=shift_x,
src/skimage/filters/rank/bilateral.py:def _apply(func, image, footprint, out, mask, shift_x, shift_y, s0, s1, out_dtype=None):
src/skimage/filters/rank/bilateral.py:        shift_x=shift_x,
src/skimage/filters/rank/bilateral.py:        shift_x=shift_x,
src/skimage/filters/rank/bilateral.py:    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, s0=10, s1=10
src/skimage/filters/rank/bilateral.py:    shift_x, shift_y : int
src/skimage/filters/rank/bilateral.py:        shift_x=shift_x,
src/skimage/filters/rank/bilateral.py:    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, s0=10, s1=10
src/skimage/filters/rank/bilateral.py:    shift_x, shift_y : int
src/skimage/filters/rank/bilateral.py:        shift_x=shift_x,
src/skimage/filters/rank/bilateral.py:    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, s0=10, s1=10
src/skimage/filters/rank/bilateral.py:    shift_x, shift_y : int
src/skimage/filters/rank/bilateral.py:        shift_x=shift_x,
src/skimage/filters/rank/percentile_cy.pyx:               signed char shift_x, signed char shift_y, cnp.float64_t p0, cnp.float64_t p1,
src/skimage/filters/rank/percentile_cy.pyx:          shift_x, shift_y, p0, p1, 0, 0, n_bins)
src/skimage/filters/rank/percentile_cy.pyx:              signed char shift_x, signed char shift_y, cnp.float64_t p0, cnp.float64_t p1,
src/skimage/filters/rank/percentile_cy.pyx:          shift_x, shift_y, p0, p1, 0, 0, n_bins)
src/skimage/filters/rank/percentile_cy.pyx:          signed char shift_x, signed char shift_y, cnp.float64_t p0, cnp.float64_t p1,
src/skimage/filters/rank/percentile_cy.pyx:          shift_x, shift_y, p0, p1, 0, 0, n_bins)
src/skimage/filters/rank/percentile_cy.pyx:         signed char shift_x, signed char shift_y, cnp.float64_t p0, cnp.float64_t p1,
src/skimage/filters/rank/percentile_cy.pyx:          shift_x, shift_y, p0, p1, 0, 0, n_bins)
src/skimage/filters/rank/percentile_cy.pyx:                   signed char shift_x, signed char shift_y, cnp.float64_t p0, cnp.float64_t p1,
src/skimage/filters/rank/percentile_cy.pyx:          out, shift_x, shift_y, p0, p1, 0, 0, n_bins)
src/skimage/filters/rank/percentile_cy.pyx:                      signed char shift_x, signed char shift_y, cnp.float64_t p0, cnp.float64_t p1,
src/skimage/filters/rank/percentile_cy.pyx:          mask, out, shift_x, shift_y, p0, p1, 0, 0, n_bins)
src/skimage/filters/rank/percentile_cy.pyx:                signed char shift_x, signed char shift_y, cnp.float64_t p0, cnp.float64_t p1,
src/skimage/filters/rank/percentile_cy.pyx:          shift_x, shift_y, p0, 1, 0, 0, n_bins)
src/skimage/filters/rank/percentile_cy.pyx:         signed char shift_x, signed char shift_y, cnp.float64_t p0, cnp.float64_t p1,
src/skimage/filters/rank/percentile_cy.pyx:          shift_x, shift_y, p0, p1, 0, 0, n_bins)
src/skimage/filters/rank/percentile_cy.pyx:               signed char shift_x, signed char shift_y, cnp.float64_t p0, cnp.float64_t p1,
src/skimage/filters/rank/percentile_cy.pyx:          shift_x, shift_y, p0, 1, 0, 0, n_bins)
src/skimage/filters/rank/core_cy_3d.pxd:                   signed char shift_x, signed char shift_y, signed char shift_z,
tests/skimage/filters/rank/test_rank.py:                image=image8, footprint=elem, mask=mask, out=out8, shift_x=0, shift_y=0
tests/skimage/filters/rank/test_rank.py:                shift_x=+1,
tests/skimage/filters/rank/test_rank.py:                image=image8, footprint=elem, mask=mask, out=out8, shift_x=0, shift_y=0
tests/skimage/filters/rank/test_rank.py:                shift_x=+1,
tests/skimage/filters/rank/test_rank.py:                shift_x=0,
tests/skimage/filters/rank/test_rank.py:                shift_x=+1,
tests/skimage/filters/rank/test_rank.py:                shift_x=0,
tests/skimage/filters/rank/test_rank.py:                shift_x=+1,
tests/skimage/filters/rank/test_rank.py:                shift_x=0,
tests/skimage/filters/rank/test_rank.py:                shift_x=+1,
tests/skimage/filters/rank/test_rank.py:                    shift_x=0,
tests/skimage/filters/rank/test_rank.py:            image=image, footprint=elem, out=out, mask=mask, shift_x=1, shift_y=1
tests/skimage/filters/rank/test_rank.py:            image=image, footprint=elem, out=out, mask=mask, shift_x=1, shift_y=1
tests/skimage/filters/rank/test_rank.py:        rank.mean(image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0)
tests/skimage/filters/rank/test_rank.py:            image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0
tests/skimage/filters/rank/test_rank.py:            image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0
tests/skimage/filters/rank/test_rank.py:            image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0
tests/skimage/filters/rank/test_rank.py:        rank.mean(image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0)
tests/skimage/filters/rank/test_rank.py:            image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0
tests/skimage/filters/rank/test_rank.py:            image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0
tests/skimage/filters/rank/test_rank.py:            image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0
tests/skimage/filters/rank/test_rank.py:        rank.mean(image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0)
tests/skimage/filters/rank/test_rank.py:            image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0
tests/skimage/filters/rank/test_rank.py:            image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0
tests/skimage/filters/rank/test_rank.py:        rank.mean(image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0)
tests/skimage/filters/rank/test_rank.py:            image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0
tests/skimage/filters/rank/test_rank.py:            image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0
tests/skimage/filters/rank/test_rank.py:            image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0
tests/skimage/filters/rank/test_rank.py:        rank.mean(image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0)
tests/skimage/filters/rank/test_rank.py:            image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0
tests/skimage/filters/rank/test_rank.py:            image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0
tests/skimage/filters/rank/test_rank.py:            image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0
tests/skimage/filters/rank/test_rank.py:                image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0
tests/skimage/filters/rank/test_rank.py:                image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0
tests/skimage/filters/rank/test_rank.py:                image=image, footprint=elem, out=out, mask=mask, shift_x=0, shift_y=0
tests/skimage/filters/rank/test_rank.py:    @pytest.mark.parametrize("shift_name", ["shift_x", "shift_y"])
tests/skimage/filters/rank/test_rank.py:    @pytest.mark.parametrize("shift_name", ["shift_x", "shift_y", "shift_z"])
src/skimage/filters/rank/core_cy.pyx:                signed char shift_x, signed char shift_y,
src/skimage/filters/rank/core_cy.pyx:    cdef Py_ssize_t centre_c = <Py_ssize_t>(footprint.shape[1] / 2) + shift_x
src/skimage/filters/rank/generic_cy.pyx:               signed char shift_x, signed char shift_y, Py_ssize_t n_bins):
src/skimage/filters/rank/generic_cy.pyx:          shift_x, shift_y, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:                  signed char shift_x, signed char shift_y, signed char shift_z,
src/skimage/filters/rank/generic_cy.pyx:             out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:              signed char shift_x, signed char shift_y, Py_ssize_t n_bins):
src/skimage/filters/rank/generic_cy.pyx:          shift_x, shift_y, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:                 signed char shift_x, signed char shift_y, signed char shift_z,
src/skimage/filters/rank/generic_cy.pyx:             out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:              signed char shift_x, signed char shift_y, Py_ssize_t n_bins):
src/skimage/filters/rank/generic_cy.pyx:          shift_x, shift_y, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:                 signed char shift_x, signed char shift_y, signed char shift_z,
src/skimage/filters/rank/generic_cy.pyx:             out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:             signed char shift_x, signed char shift_y, Py_ssize_t n_bins):
src/skimage/filters/rank/generic_cy.pyx:          shift_x, shift_y, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:                signed char shift_x, signed char shift_y, signed char shift_z,
src/skimage/filters/rank/generic_cy.pyx:             out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:          signed char shift_x, signed char shift_y, Py_ssize_t n_bins):
src/skimage/filters/rank/generic_cy.pyx:          shift_x, shift_y, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:             signed char shift_x, signed char shift_y, signed char shift_z,
src/skimage/filters/rank/generic_cy.pyx:             shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:                    signed char shift_x, signed char shift_y, Py_ssize_t n_bins):
src/skimage/filters/rank/generic_cy.pyx:          out, shift_x, shift_y, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:                       signed char shift_x, signed char shift_y, signed char shift_z,
src/skimage/filters/rank/generic_cy.pyx:             mask, out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:                   signed char shift_x, signed char shift_y, Py_ssize_t n_bins):
src/skimage/filters/rank/generic_cy.pyx:          out, shift_x, shift_y, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:                      signed char shift_x, signed char shift_y, signed char shift_z,
src/skimage/filters/rank/generic_cy.pyx:             mask, out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:            signed char shift_x, signed char shift_y, Py_ssize_t n_bins):
src/skimage/filters/rank/generic_cy.pyx:          shift_x, shift_y, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:               signed char shift_x, signed char shift_y, signed char shift_z,
src/skimage/filters/rank/generic_cy.pyx:             shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:             signed char shift_x, signed char shift_y, Py_ssize_t n_bins):
src/skimage/filters/rank/generic_cy.pyx:          shift_x, shift_y, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:                signed char shift_x, signed char shift_y, signed char shift_z,
src/skimage/filters/rank/generic_cy.pyx:             out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:                      signed char shift_x, signed char shift_y, Py_ssize_t n_bins):
src/skimage/filters/rank/generic_cy.pyx:          out, shift_x, shift_y, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:                         signed char shift_x, signed char shift_y, signed char shift_z,
src/skimage/filters/rank/generic_cy.pyx:             mask, out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:           signed char shift_x, signed char shift_y, Py_ssize_t n_bins):
src/skimage/filters/rank/generic_cy.pyx:          shift_x, shift_y, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:              signed char shift_x, signed char shift_y, signed char shift_z,
src/skimage/filters/rank/generic_cy.pyx:             shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:         signed char shift_x, signed char shift_y, Py_ssize_t n_bins):
src/skimage/filters/rank/generic_cy.pyx:          shift_x, shift_y, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:            signed char shift_x, signed char shift_y, signed char shift_z,
src/skimage/filters/rank/generic_cy.pyx:             shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:         signed char shift_x, signed char shift_y, Py_ssize_t n_bins):
src/skimage/filters/rank/generic_cy.pyx:          out, shift_x, shift_y, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:            signed char shift_x, signed char shift_y, signed char shift_z,
src/skimage/filters/rank/generic_cy.pyx:             shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:               signed char shift_x, signed char shift_y, Py_ssize_t n_bins):
src/skimage/filters/rank/generic_cy.pyx:          shift_x, shift_y, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:                  signed char shift_x, signed char shift_y, signed char shift_z,
src/skimage/filters/rank/generic_cy.pyx:             out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:                  signed char shift_x, signed char shift_y, Py_ssize_t n_bins):
src/skimage/filters/rank/generic_cy.pyx:          out, shift_x, shift_y, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:                     signed char shift_x, signed char shift_y, signed char shift_z,
src/skimage/filters/rank/generic_cy.pyx:             mask, out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:             signed char shift_x, signed char shift_y, Py_ssize_t n_bins):
src/skimage/filters/rank/generic_cy.pyx:          shift_x, shift_y, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:                signed char shift_x, signed char shift_y, signed char shift_z,
src/skimage/filters/rank/generic_cy.pyx:             out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:          signed char shift_x, signed char shift_y, Py_ssize_t n_bins):
src/skimage/filters/rank/generic_cy.pyx:          shift_x, shift_y, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:             signed char shift_x, signed char shift_y, signed char shift_z,
src/skimage/filters/rank/generic_cy.pyx:             shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:                   signed char shift_x, signed char shift_y, Py_ssize_t n_bins):
src/skimage/filters/rank/generic_cy.pyx:          shift_x, shift_y, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:              signed char shift_x, signed char shift_y, Py_ssize_t n_bins):
src/skimage/filters/rank/generic_cy.pyx:          shift_x, shift_y, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/generic_cy.pyx:                 signed char shift_x, signed char shift_y, signed char shift_z,
src/skimage/filters/rank/generic_cy.pyx:             out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)
src/skimage/filters/rank/core_cy_3d.pyx:                   signed char shift_x, signed char shift_y,
src/skimage/filters/rank/core_cy_3d.pyx:    cdef Py_ssize_t centre_p = (footprint.shape[0] // 2) + shift_x
src/skimage/filters/rank/core_cy_3d.pyx:            "half footprint + shift_x must be between 0 and footprint"
doc/source/release_notes/release_0.23.rst:- Parameters ``shift_x`` and ``shift_y`` in ``skimage.morphology.erosion`` and ``skimage.morphology.dilation`` are deprecated. Use ``pad_footprint`` or modify the footprint manually instead (`#6695 <https://github.com/scikit-image/scikit-image/pull/6695>`_).
doc/source/release_notes/release_0.23.rst:- Change the default value of the parameters ``shift_x``, ``shift_y`` and ``shift_z`` from ``False`` to ``0`` in the ``skimage.filters.rank`` functions. This has not impact on the  results. Warn in case boolean shifts are provided from now on (`#7320 <https://github.com/scikit-image/scikit-image/pull/7320>`_).
doc/source/release_notes/release_0.17.rst:   * ``mask``, ``shift_x``, and ``shift_y`` from ``skimage.filters.median``
doc/source/release_notes/release_0.15.rst:  behavior (i.e., ``mask``, ``shift_x``, ``shift_y``) will be removed.
```
