# Coordinate conventions in `skimage.transform` and related code

Summary of findings on **ij** (Numpy) vs **xy** (imaging) coordinate usage across `skimage.transform` and closely related code, as requested in `GEMINI.md`. Evidence is given with code snippets and links to the [scikit-image GitHub repository](https://github.com/scikit-image/scikit-image).

**Conventions (recap from GEMINI.md):**

- **ij (Numpy):** `(p, q)` → `img[p, q]`. Row `r` = `img[r, :]`, column `c` = `img[:, c]`. First index = first axis, second = second axis.
- **xy (imaging):** `(p, q)` → `img[q, p]`. First coordinate = position on second axis (column), second = first axis (row). Often described as x = column, y = row.

**References:** [Issue #7728](https://github.com/scikit-image/scikit-image/issues/7728) (ij in `skimage.transform`), [Issue #2275](https://github.com/scikit-image/scikit-image/issues/2275) (xy/rc conversion), [User guide — Coordinate conventions](https://scikit-image.org/docs/stable/user_guide/numpy_images.html#coordinate-conventions).

---

## 1. Warping and geometric transforms

### 1.1 `warp` and `_warp_fast` (Cython)

**Convention: xy**

**Evidence:**

- In [`_warps_cy.pyx`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_warps_cy.pyx), the output loop iterates over `(tfr, tfc)` (output row, col). The inverse map is called with `(tfc, tfr)` as `(x, y)`:

```python
for tfr in range(out_r):
    for tfc in range(out_c):
        transform_func(tfc, tfr, &M[0, 0], &c, &r)
        interp_func(&img[0, 0], rows, cols, r, c, ...)
```

- So output coordinates passed to the transform are **(x, y) = (column, row)**. The transform returns source `(c, r)` (col, row), and the image is sampled at `img[r, c]` (row, col). The _meaning_ of coordinates in the transform chain is xy.

**Callers:** `rotate`, `swirl`, `radon`, `warp_polar`, geometric transforms used as `inverse_map`. Examples: [`doc/examples/transform/plot_geometric.py`](https://github.com/scikit-image/scikit-image/blob/main/doc/examples/transform/plot_geometric.py) uses `warp` with `SimilarityTransform`; [`tests/.../test_warps.py`](https://github.com/scikit-image/scikit-image/blob/main/tests/skimage/transform/test_warps.py) uses `warp` with `AffineTransform` / `ProjectiveTransform`.

---

### 1.2 `warp_coords`

**Convention: xy (user-facing coord_map)**

**Evidence:**

- Docstring says coordinates are “(row, col) pairs”, but the implementation builds a grid with **columns first**:

```python
# _warps.py
tf_coords = np.indices((cols, rows), dtype=dtype).reshape(2, -1).T  # (col, row) order
tf_coords = coord_map(tf_coords)
# ...
coords[1, ...] = tf_coords[0, ...]  # map_coords axis 1
coords[0, ...] = tf_coords[1, ...]  # map_coords axis 0
```

- The example uses `xy` and a shift `[-20, 10]` (x, y):

```python
def shift_up10_left20(xy):
    return xy - np.array([-20, 10])[None, :]
```

- So the **coord_map** receives and returns **(x, y) = (col, row)**. The result is then rearranged so that `map_coordinates` gets (axis0, axis1) = (row, col).

**Code:** [`src/skimage/transform/_warps.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_warps.py) — `warp_coords`, and the `warp` branch that uses it (e.g. when `inverse_map` is a callable).

---

### 1.3 `rotate`

**Convention: xy**

**Evidence:**

- Docstring explicitly states the **center** is “(cols, rows), contrary to normal skimage ordering”:

```python
# center=(cols / 2 - 0.5, rows / 2 - 0.5)
```

- Default center:

```python
center = np.array((cols, rows)) / 2.0 - 0.5
```

- Corners for resize are **(col, row)**:

```python
corners = np.array([[0, 0], [0, rows - 1], [cols - 1, rows - 1], [cols - 1, 0]])
corners = tform.inverse(corners)
minc, minr = corners[:, 0].min(), corners[:, 1].min()
```

**Code:** [`src/skimage/transform/_warps.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_warps.py) — `rotate`.

---

### 1.4 `swirl`

**Convention: xy**

**Evidence:**

- Docstring: `center` is “(column, row) tuple”. Default uses `image.shape[:2][::-1]` (cols, rows).

**Code:** [`src/skimage/transform/_warps.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_warps.py) — `swirl`.

---

### 1.5 `_linear_polar_mapping` / `_log_polar_mapping` (used by `warp_polar`)

**Convention: xy for output coords; center as (row, col) = ij**

**Evidence:**

- Docstring: `output_coords` are “(col, row)”; `center` is “tuple (row, col)”.
- Implementation uses `center[0]` for radial offset in the first axis and `center[1]` for the second, consistent with (row, col) for center.

**Code:** [`src/skimage/transform/_warps.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_warps.py) — `_linear_polar_mapping`, `_log_polar_mapping`.

---

### 1.6 Geometric transform classes (`ProjectiveTransform`, `AffineTransform`, `SimilarityTransform`, etc.)

**Convention: xy**

**Evidence:**

- Used as `inverse_map` in `warp` and `_warp_fast`, which treat coords as (x, y) = (col, row).
- `rotate` builds `SimilarityTransform(translation=center)` with `center = (cols, rows)`.
- `ProjectiveTransform` docstring describes the map as `X = (a0*x + a1*y + a2) / (...)`, with `x, y` as first and second components; `PolynomialTransform.__call__` uses `x = coords[:, 0]`, `y = coords[:, 1]` in the same way.
- In [`plot_geometric.py`](https://github.com/scikit-image/scikit-image/blob/main/doc/examples/transform/plot_geometric.py), `src`/`dst` are rectangle corners in what is effectively (x, y) layout for estimation.

**Code:** [`src/skimage/transform/_geometric.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_geometric.py).

---

## 2. Hough transforms

### 2.1 `hough_line`

**Convention: xy**

**Evidence:**

- Notes: “The origin is the top left corner. X and Y axis are horizontal and vertical edges respectively.”
- Cython `_hough_line`: `y_idxs, x_idxs = np.nonzero(img)`, so **x = col, y = row**. Distance uses `ctheta[j] * x + stheta[j] * y` (standard Hough (x,y) form). Image indexing is `img[row, col]` elsewhere.

**Code:** [`hough_transform.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/hough_transform.py), [`_hough_transform.pyx`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_hough_transform.pyx).

---

### 2.2 `hough_line_peaks`

**Convention: ij for input image; Hough space is (distances, angles)**

**Evidence:**

- Example uses `draw.line` and `img[rr, cc]` (ij):

```python
rr, cc = line(0, 0, 14, 14)
img[rr, cc] = 1
```

- Second line uses `img[cc, rr]` — **swapped** relative to `draw.line`’s (rr, cc). That is likely a bug; `draw.line` returns (row, col) and expects `img[rr, cc]`.

**Code:** [`hough_transform.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/hough_transform.py).

---

### 2.3 `probabilistic_hough_line`

**Convention: xy**

**Evidence:**

- Returns “lines in format `((x0, y0), (x1, y1))`”.
- Cython: `y_idxs, x_idxs = np.nonzero(img)`, `mask[y_idxs, x_idxs] = 1`; line endpoints stored as `(x, y)`; bounds use `width` (x) and `height` (y).
- Example [`plot_line_hough_transform.py`](https://github.com/scikit-image/scikit-image/blob/main/doc/examples/edges/plot_line_hough_transform.py): `plot((p0[0], p1[0]), (p0[1], p1[1]))` with `set_xlim(0, shape[1])`, `set_ylim(shape[0], 0)`, i.e. (x, y) = (col, row).

**Code:** [`hough_transform.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/hough_transform.py), [`_hough_transform.pyx`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_hough_transform.pyx).

---

### 2.4 `hough_circle`

**Convention: ij**

**Evidence:**

- Example: `circle_perimeter(25, 35, 23)` → `rr, cc`; `img[rr, cc] = 1`. Unravel gives `(ridx, r, c)` and the example uses `(r, c)` as (row, col) for the center. So **output (r, c) = (row, col)**.
- Cython uses `x, y = np.nonzero(img)` but assigns **rows to `x` and cols to `y`** (names reversed). Accumulator is `(radii, rows, cols)` and indexed as `acc[i, tx, ty]` with `tx, ty` derived from row/col. The **effective** convention for the public API is ij.

**Code:** [`hough_transform.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/hough_transform.py), [`_hough_transform.pyx`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_hough_transform.pyx).

---

### 2.5 `hough_circle_peaks`

**Convention: xy**

**Evidence:**

- Returns “accum, cx, cy, rad” with “x and y center coordinates”.
- Uses `_prominent_peaks`, which returns `(intensity, xcoords, ycoords)` with **x = axis 1 (col), y = axis 0 (row)**.
- Example in docstring: `draw.circle_perimeter(y_0, x_0, radius)` → `y, x`; `img[x, y] = 1` → **img[col, row]**.
- Example [`plot_circular_elliptical_hough_transform.py`](https://github.com/scikit-image/scikit-image/blob/main/doc/examples/edges/plot_circular_elliptical_hough_transform.py): `zip(cy, cx, radii)`, `circle_perimeter(center_y, center_x, ...)`, so **cy = row, cx = col** (xy).

**Inconsistency:** `hough_circle` returns (r, c) = (row, col) (ij); `hough_circle_peaks` returns (cx, cy) = (col, row) (xy). Converting between them requires swapping.

**Code:** [`hough_transform.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/hough_transform.py); [`feature/peak.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/peak.py) — `_prominent_peaks`.

---

### 2.6 `hough_ellipse`

**Convention: xy**

**Evidence:**

- Returns “(yc, xc)” for center; Cython uses `p1x = pixels[1, p1]` (col), `p1y = pixels[0, p1]` (row).
- Example: `ellipse_perimeter(10, 10, 6, 8)` → `rr, cc`; `img[cc, rr] = 1` → **img[col, row]**.
- Ellipse example uses `yc, xc` from the result.

**Code:** [`hough_transform.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/hough_transform.py), [`_hough_transform.pyx`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_hough_transform.pyx).

---

## 3. Radon transform

### 3.1 `radon` / `iradon`

**Convention: ij for image axes; warp uses xy**

**Evidence:**

- Image center: “pixel with indices `(image.shape[0] // 2, image.shape[1] // 2)`” (row, col).
- Rotation is applied via `warp(padded_image, R, ...)`, so the **matrix** is in xy. `radon_image[:, i] = rotated.sum(0)` sums over the first image axis (rows).

**Code:** [`radon_transform.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/radon_transform.py).

---

### 3.2 `_radon_transform` Cython (bilinear ray sum)

**Convention: xy-style (x,y) mapped to image[i,j]**

**Evidence:**

- Comment “(s, t) is the (x, y) system rotated by theta”; `index_i = x + rotation_center`, `index_j = y + rotation_center`; `image[i, j]` with i = first axis, j = second. So (x,y) → (row, col) in typical usage.

**Code:** [`_radon_transform.pyx`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_radon_transform.pyx).

---

## 4. Resize, rescale, downscale, integral

### 4.1 `resize`, `rescale`, `downscale_local_mean`, `resize_local_mean`

**Convention: ij (shape only)**

**Evidence:**

- `output_shape` documented as “(rows, cols[, ...][, dim])”. No point-wise coordinate API; only array shapes, which follow NumPy (ij).

**Code:** [`_warps.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_warps.py), etc.

---

### 4.2 `integral_image`, `integrate`

**Convention: ij**

**Evidence:**

- `integral_image`: `S[m, n] = sum over i<=m, j<=n` of `X[i, j]` — standard (row, col) indexing.
- `integrate`: “starting row, col, …” / “end row, col, …” for window corners.

**Code:** [`integral.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/integral.py).

---

## 5. Pyramids, finite Radon, thin-plate splines

- **Pyramids** (`pyramid_*`): operate on arrays only; no explicit coordinate convention.
- **`frt2` / `ifrt2`**: finite Radon; similar to radon, image axes are ij.
- **`ThinPlateSplineTransform`**: used with `from_estimate(src, dst)` and `warp`; **src/dst** are typically in the same (x, y) style as other geometric transforms (xy).

---

## 6. Supporting code outside `transform`

### 6.1 `skimage.draw` (`line`, `circle_perimeter`, `ellipse_perimeter`)

**Convention: ij**

**Evidence:**

- `line(r0, c0, r1, c1)`, `circle_perimeter(r, c, radius)`, etc. Return `(rr, cc)` for `img[rr, cc]`.

**Code:** [`draw/draw.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/draw/draw.py).

---

### 6.2 `skimage.feature.corner_subpix`

**Convention: ij**

**Evidence:**

- Docstring: “Corner coordinates `(row, col)`.” Loop uses `(y0, x0)` for (row, col) and `image[miny:maxy, minx:maxx]`.

**Code:** [`feature/corner.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/corner.py).

---

### 6.3 `skimage.feature._prominent_peaks`

**Convention: xy**

**Evidence:**

- Returns “intensity, xcoords, ycoords” with **x = axis 1 (col), y = axis 0 (row)**. Used by `hough_line_peaks` and `hough_circle_peaks`.

**Code:** [`feature/peak.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/peak.py).

---

## 7. Summary table

| Function / API                | Convention | Notes                                    |
| ----------------------------- | ---------- | ---------------------------------------- |
| `warp`, `_warp_fast`          | xy         | (x,y) = (col, row) in transform          |
| `warp_coords`                 | xy         | coord_map (x,y)                          |
| `rotate`, `swirl`             | xy         | center (col, row)                        |
| `warp_polar` mappings         | xy / ij    | output (col, row); center (row, col)     |
| Geometric transforms          | xy         | Used with warp                           |
| `hough_line`                  | xy         | x=col, y=row                             |
| `hough_line_peaks`            | ij         | Example uses img[rr,cc]; second line bug |
| `probabilistic_hough_line`    | xy         | ((x0,y0),(x1,y1))                        |
| `hough_circle`                | ij         | (r, c) = (row, col)                      |
| `hough_circle_peaks`          | xy         | (cx, cy) = (col, row)                    |
| `hough_ellipse`               | xy         | (yc, xc); img[cc,rr] in example          |
| `radon` / `iradon`            | ij + xy    | Image ij; warp xy                        |
| `resize`, `rescale`, etc.     | ij         | shape (rows, cols)                       |
| `integral_image`, `integrate` | ij         | row, col                                 |
| `draw.line`, etc.             | ij         | (rr, cc) → img[rr,cc]                    |
| `corner_subpix`               | ij         | (row, col)                               |
| `_prominent_peaks`            | xy         | x=col, y=row                             |

---

## 8. Recommendations for issue #7728

1. **Add an explicit coordinate/convention flag** (e.g. `coordinates='xy'|'ij'`) to warps, `rotate`, `swirl`, `warp_polar`, and geometric transforms, as in #7728.
2. **Unify Hough API:** e.g. have `hough_circle` and `hough_circle_peaks` use the same convention (preferably ij to match `draw` and NumPy), and fix the `hough_line_peaks` example (`img[cc, rr]`).
3. **Clarify docstrings** wherever “x”/“y” or “row”/“col” are used, and point to the user guide’s coordinate conventions.
4. **Consider the conversion helpers** from #2275 (e.g. xy ↔ rc) to support interoperability with OpenCV and other libraries.

---

_Generated from the `cursor-coordinate-review` branch of scikit-image. Base links target the default branch on GitHub._
