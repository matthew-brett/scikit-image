"""Test shift values of rank filters."""

import inspect

import numpy as np
import pytest

from skimage.morphology import disk
from skimage.filters import rank
from skimage.filters.rank import __all__ as all_rank_filters
from skimage._shared.testing import assert_equal

ported_rank_filters = ['mean_bilateral']
unported_rank_filters = set(all_rank_filters).difference(ported_rank_filters)


@pytest.fixture(scope="module")
def rand_img():
    np.random.seed(0)
    return np.random.randint(0, 250, size=(45, 47), dtype=np.uint8)


@pytest.fixture(scope="module")
def disk5():
    return disk(5)


@pytest.mark.parametrize("filt_name", unported_rank_filters)
def test_unported_params(filt_name, rand_img, disk5):
    filt = getattr(rank, filt_name)
    out_params = list(inspect.signature(filt).parameters)
    assert out_params[:5] == ['image', 'footprint', 'out', 'mask', 'shift_x']
    out = filt(rand_img, disk5)
    assert_equal(out, filt(rand_img, disk5, shift_x=0, shift_y=0))


@pytest.mark.parametrize("filt_name", ported_rank_filters)
def test_params(filt_name, rand_img, disk5):
    filt = getattr(rank, filt_name)
    out_params = list(inspect.signature(filt).parameters)
    assert out_params[:5] == ['image', 'footprint', 'out', 'mask', 'shift']
    out = filt(rand_img, disk5)
    # New convention, default values.
    assert_equal(out, filt(rand_img, disk5, shift=(0, 0)))
    # Old convention, default values, raises warning.
    with pytest.warns(FutureWarning, match='`shift_x` and `shift_y` are deprecated'):
        assert_equal(out, filt(rand_img, disk5, shift_x=0, shift_y=0))
    with pytest.raises(ValueError, match='Cannot mix `shift` and either `shift_x`'):
        filt(rand_img, disk5, shift=(0, 0), shift_x=0)
    # New positional only warnings.
    with pytest.warns(
        FutureWarning,
        match=r'All \*positional\* arguments to '
        f'`{filt.__name__}` after `footprint` are deprecated',
    ):
        assert_equal(out, filt(rand_img, disk5, None))
    # Equivalence of old and new conventions.
    out_23 = filt(rand_img, disk5, shift=(2, 3))
    with pytest.warns(FutureWarning, match='`shift_x` and `shift_y` are deprecated'):
        assert_equal(out_23, filt(rand_img, disk5, shift_x=3, shift_y=2))
    with pytest.warns(FutureWarning, match='`shift_x` and `shift_y` are deprecated'):
        assert_equal(
            filt(rand_img, disk5, shift=(0, 3)), filt(rand_img, disk5, shift_x=3)
        )
    with pytest.warns(FutureWarning, match='`shift_x` and `shift_y` are deprecated'):
        assert_equal(
            filt(rand_img, disk5, shift=(2, 0)), filt(rand_img, disk5, shift_y=2)
        )


@pytest.mark.parametrize("filt_name", ported_rank_filters)
def test_unwrapped_params(filt_name, rand_img, disk5):
    # Check unwrapped function raises error on shift_x, shift_y
    filt = getattr(rank, filt_name)
    out = filt(rand_img, disk5)
    filt2 = rank.bilateral.unwrap_2to1(filt)
    with pytest.raises(TypeError):
        filt2(rand_img, disk5, shift_x=0, shift_y=0)
    assert_equal(out, filt2(rand_img, disk5, shift=(0, 0)))
