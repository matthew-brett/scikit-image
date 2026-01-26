# Coordinate Convention Analysis for scikit-image

Based on investigation of the scikit-image codebase, here is a comprehensive summary of coordinate conventions used across the codebase.

## Summary Table

| Module/Function                                             | Convention        | Evidence                                                             |
| ----------------------------------------------------------- | ----------------- | -------------------------------------------------------------------- |
| **skimage.draw** (all functions)                            | **IJ (row, col)** | Returns `(rr, cc)` for `img[rr, cc]`                                 |
| **skimage.feature** (corner*peaks, peak_local_max, blob*\*) | **IJ (row, col)** | Returns `(row, col)` pairs                                           |
| **skimage.measure** (find_contours, regionprops)            | **IJ (row, col)** | Coordinates used as `array[r, c]`                                    |
| **skimage.transform** (geometric transforms)                | **XY**            | Math uses x, y; coords are (x, y) pairs                              |
| **skimage.transform.rotate**                                | **XY for center** | Explicitly notes "(cols, rows), contrary to normal skimage ordering" |
| **skimage.transform.hough_ellipse**                         | **MIXED/BUGGY**   | Returns (yc, xc) but examples use `img[cc, rr]`                      |
| **skimage.filters.rank** (shift_x, shift_y params)          | **XY**            | `shift_x` shifts columns (j), `shift_y` shifts rows (i)              |

## Critical Findings

### 1. Bug in hough_line_peaks docstring example (lines 55-58)

**File**: [`src/skimage/transform/hough_transform.py:55-58`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/hough_transform.py#L55-L58)

```python
>>> rr, cc = line(0, 0, 14, 14)
>>> img[rr, cc] = 1          # Correct: ij convention
>>> rr, cc = line(0, 14, 14, 0)
>>> img[cc, rr] = 1          # BUG: swapped to cc, rr!
```

This inconsistency within the same example is clearly a bug.

### 2. hough_ellipse uses mixed conventions (lines 148-162)

**File**: [`src/skimage/transform/hough_transform.py:148-162`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/hough_transform.py#L148-L162)

```python
# Returns: (accumulator, yc, xc, a, b, orientation) - note yc before xc
>>> rr, cc = ellipse_perimeter(10, 10, 6, 8)
>>> img[cc, rr] = 1  # SWAPPED! Uses cc as first index
```

The return format uses (yc, xc) which is neither standard ij nor xy convention.

### 3. rotate() explicitly acknowledges inconsistency (lines 369-373)

**File**: [`src/skimage/transform/_warps.py:369-373`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_warps.py#L369-L373)

```python
center : iterable of length 2
    The rotation center. If ``center=None``, the image is rotated around
    its center, i.e. ``center=(cols / 2 - 0.5, rows / 2 - 0.5)``.  Please
    note that this parameter is (cols, rows), contrary to normal skimage
    ordering.
```

## Module-by-Module Details

### skimage.draw — IJ Convention

All draw functions use `(r, c)` parameters and return `(rr, cc)` for direct array indexing.

**Example from [`src/skimage/draw/draw.py:363-378`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/draw/draw.py#L363-L378)**:

```python
def line(r0, c0, r1, c1):
    """
    r0, c0 : int
        Starting position (row, column).
    Returns
    -------
    rr, cc : (N,) ndarray of int
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.
    """
```

### skimage.feature — IJ Convention

**corner_peaks** from [`src/skimage/feature/corner.py:1163`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/corner.py#L1163):

```python
"* If `indices = True`  : (row, column, ...) coordinates of peaks."
```

**blob_dog/log/doh** from [`src/skimage/feature/blob.py:286-287`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/blob.py#L286-L287):

```python
"(r, c, sigma)" or "(p, r, c, sigma)" where "(r, c)" or
"(p, r, c)" are coordinates of the blob
```

### skimage.measure — IJ Convention

**find_contours** from [`src/skimage/measure/_find_contours.py:47`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/measure/_find_contours.py#L47):

```python
"contours : list of (K, 2) ndarrays
Each contour is a ndarray of ``(row, column)`` coordinates along the contour."
```

**regionprops.coords** from [`src/skimage/measure/_regionprops.py:499-502`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/measure/_regionprops.py#L499-L502):

```python
indices = np.argwhere(self.image)  # Returns (row, col) from argwhere
```

### skimage.transform (geometric) — XY Convention

**ProjectiveTransform** from [`src/skimage/transform/_geometric.py:1086-1087`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_geometric.py#L1086-L1087):

```python
X = (a0*x + a1*y + a2) / (c0*x + c1*y + 1)
Y = (b0*x + b1*y + b2) / (c0*x + c1*y + 1)
```

Test coordinates in [`tests/skimage/transform/test_geometric.py:35-57`](https://github.com/scikit-image/scikit-image/blob/main/tests/skimage/transform/test_geometric.py#L35-L57):

```python
DST = np.array([
    [0, 0],
    [0, 5800],
    [4900, 5800],
    [4900, 0],
    ...
])  # Clearly (x, y) pairs forming a rectangle
```

### skimage.filters.rank — XY Convention (for shift parameters)

The rank filters accept `shift_x` and `shift_y` parameters that offset the footprint center. These use XY convention where x=horizontal (column) and y=vertical (row).

**Implementation from [`src/skimage/filters/rank/core_cy.pyx:68-69`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/filters/rank/core_cy.pyx#L68-L69)**:

```cython
cdef Py_ssize_t centre_r = <Py_ssize_t>(footprint.shape[0] / 2) + shift_y
cdef Py_ssize_t centre_c = <Py_ssize_t>(footprint.shape[1] / 2) + shift_x
```

This shows:

- `shift_x` is added to `centre_c` (column = second axis = j direction)
- `shift_y` is added to `centre_r` (row = first axis = i direction)

The parameter names follow XY convention where x is horizontal and y is vertical, opposite to the IJ convention used elsewhere in the codebase.

## Conclusion

The codebase has a **fundamental split**:

1. **skimage.draw, skimage.feature, skimage.measure** — consistently use **IJ (row, col)** convention
2. **skimage.transform geometric transforms** — use **XY** convention for coordinate pairs
3. **skimage.filters.rank shift parameters** — use **XY** convention (`shift_x`=column, `shift_y`=row)
4. **skimage.transform hough functions** — **inconsistent/buggy**, mixing conventions even within single examples

The explicit acknowledgment in `rotate()` that its center parameter is "(cols, rows), contrary to normal skimage ordering" confirms the developers are aware of this inconsistency. The GitHub issue [#7728](https://github.com/scikit-image/scikit-image/issues/7728) aims to address this by adding a `coordinates` parameter across `skimage.transform`.
