import numpy as np
from skimage import data
from skimage.morphology import disk
from skimage.filters.rank import mean_bilateral

img = data.camera().astype(np.uint16)
# No deprecation.
bilat_img = mean_bilateral(img, disk(20))
shifted = mean_bilateral(img, disk(20), shift=(1, 3))
# Deprecation warning
print('Deprecation warning for kwargs')
assert np.all(mean_bilateral(img, disk(20), None) == bilat_img)
print('Deprecation warning 2 for kwargs')
assert np.all(mean_bilateral(img, disk(20), None, None) == bilat_img)
print('Deprecation warning 3 for kwargs')
assert np.all(
    mean_bilateral(img, disk(20), None, None, shift=(0, 0), s0=10, s1=10) == bilat_img
)

try:
    mean_bilateral(img, disk(20), None, None, shift=(0, 0), shift_x=0)
except ValueError:
    print('ValueError raised')
else:
    print('ValueError not raised')

# Deprecation warning
print('Deprecation warning for kwargs')
assert np.all(mean_bilateral(img, disk(20), None, None, (0, 0)) == bilat_img)
print('Deprecation warning shift_x')
bilat_img = mean_bilateral(img, disk(20), shift_x=0, s0=10, s1=10)
print('Deprecation warning shift_y')
bilat_img = mean_bilateral(img, disk(20), shift_y=0, s0=10, s1=10)
