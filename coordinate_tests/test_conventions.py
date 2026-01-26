"""
Minimal test scripts demonstrating coordinate conventions in scikit-image.

Each test uses asymmetric coordinates so the convention (ij vs xy) is
unambiguous from the visual output.

Convention reminder:
- IJ (row, col): first coord is row (vertical), second is col (horizontal)
  img[p, q] accesses position (p, q)
- XY: first coord is x (horizontal), second is y (vertical)
  img[q, p] accesses position (p, q)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# =============================================================================
# TEST 1: skimage.draw.line - demonstrates IJ convention
# =============================================================================
def test_draw_line():
    """
    draw.line takes (r0, c0, r1, c1) and returns (rr, cc).

    We draw from (10, 5) to (10, 80) - if IJ convention:
    - row=10 means 10 pixels from TOP
    - col=5 to 80 means horizontal line near top

    If it were XY convention, we'd see a vertical line on the left.
    """
    from skimage.draw import line

    img = np.zeros((100, 100), dtype=np.uint8)

    # Draw line from (row=10, col=5) to (row=10, col=80)
    rr, cc = line(10, 5, 10, 80)
    img[rr, cc] = 255

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap='gray')
    ax.set_title(
        'draw.line(r0=10, c0=5, r1=10, c1=80)\n'
        'IJ convention: horizontal line near TOP\n'
        '(If XY, would be vertical line on LEFT)'
    )
    ax.set_xlabel('Column (horizontal axis)')
    ax.set_ylabel('Row (vertical axis)')
    plt.tight_layout()
    plt.savefig('test_01_draw_line.png', dpi=100)
    plt.close()
    print("Saved: test_01_draw_line.png")


# =============================================================================
# TEST 2: skimage.draw.disk - demonstrates IJ convention
# =============================================================================
def test_draw_disk():
    """
    draw.disk takes center=(r, c) and returns (rr, cc).

    We draw at center (20, 70) - if IJ convention:
    - row=20 means 20 pixels from TOP
    - col=70 means 70 pixels from LEFT (near right side)

    If it were XY convention, disk would be at bottom-left.
    """
    from skimage.draw import disk

    img = np.zeros((100, 100), dtype=np.uint8)

    # Draw disk at (row=20, col=70), radius=15
    rr, cc = disk((20, 70), 15)
    img[rr, cc] = 255

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap='gray')
    ax.set_title(
        'draw.disk(center=(20, 70), radius=15)\n'
        'IJ convention: disk at TOP-RIGHT\n'
        '(If XY, would be at BOTTOM-LEFT)'
    )
    ax.set_xlabel('Column (horizontal axis)')
    ax.set_ylabel('Row (vertical axis)')
    # Mark the center point
    ax.plot(70, 20, 'r+', markersize=15, markeredgewidth=2)
    ax.annotate(
        'center=(20,70)',
        (70, 20),
        xytext=(40, 40),
        arrowprops=dict(arrowstyle='->', color='red'),
        color='red',
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig('test_02_draw_disk.png', dpi=100)
    plt.close()
    print("Saved: test_02_draw_disk.png")


# =============================================================================
# TEST 3: skimage.draw.ellipse - demonstrates IJ convention
# =============================================================================
def test_draw_ellipse():
    """
    draw.ellipse takes (r, c, r_radius, c_radius).

    We draw at center (25, 75) with r_radius=10, c_radius=20:
    - If IJ: center near top-right, wider horizontally (c_radius > r_radius)
    - If XY: center near bottom-left, taller vertically
    """
    from skimage.draw import ellipse

    img = np.zeros((100, 100), dtype=np.uint8)

    # Ellipse at (row=25, col=75), row_radius=10, col_radius=20
    rr, cc = ellipse(25, 75, 10, 20)
    # Clip to image bounds
    valid = (rr >= 0) & (rr < 100) & (cc >= 0) & (cc < 100)
    img[rr[valid], cc[valid]] = 255

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap='gray')
    ax.set_title(
        'draw.ellipse(r=25, c=75, r_radius=10, c_radius=20)\n'
        'IJ convention: ellipse at TOP-RIGHT, wider horizontally\n'
        '(If XY, would be at BOTTOM-LEFT, taller vertically)'
    )
    ax.set_xlabel('Column (horizontal axis)')
    ax.set_ylabel('Row (vertical axis)')
    ax.plot(75, 25, 'r+', markersize=15, markeredgewidth=2)
    plt.tight_layout()
    plt.savefig('test_03_draw_ellipse.png', dpi=100)
    plt.close()
    print("Saved: test_03_draw_ellipse.png")


# =============================================================================
# TEST 4: skimage.feature.peak_local_max - demonstrates IJ convention
# =============================================================================
def test_feature_peak_local_max():
    """
    peak_local_max returns coordinates as (row, col) pairs.

    We create a peak at position [15, 80] in the array.
    If IJ convention, returned coords should be (15, 80).
    """
    from skimage.feature import peak_local_max

    img = np.zeros((100, 100), dtype=np.float64)

    # Create a bright spot at row=15, col=80
    img[15, 80] = 1.0
    # Add some smoothing to make it a proper peak
    from scipy.ndimage import gaussian_filter

    img = gaussian_filter(img, sigma=3)

    # Find peaks
    coords = peak_local_max(img, min_distance=5, num_peaks=1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap='hot')

    # Plot found peak
    if len(coords) > 0:
        r, c = coords[0]
        ax.plot(c, r, 'go', markersize=15, markerfacecolor='none', markeredgewidth=2)
        ax.annotate(
            f'Found: ({r}, {c})',
            (c, r),
            xytext=(c - 30, r + 20),
            color='green',
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='green'),
        )

    ax.set_title(
        f'peak_local_max returns: {coords[0] if len(coords) > 0 else "none"}\n'
        'Peak placed at img[15, 80]\n'
        'IJ convention: returns (row=15, col=80) -> TOP-RIGHT'
    )
    ax.set_xlabel('Column (horizontal axis)')
    ax.set_ylabel('Row (vertical axis)')
    plt.tight_layout()
    plt.savefig('test_04_peak_local_max.png', dpi=100)
    plt.close()
    print("Saved: test_04_peak_local_max.png")


# =============================================================================
# TEST 5: skimage.feature.blob_dog - demonstrates IJ convention
# =============================================================================
def test_feature_blob_dog():
    """
    blob_dog returns (r, c, sigma) tuples.

    We create a blob at row=20, col=75.
    """
    from skimage.feature import blob_dog
    from skimage.draw import disk

    img = np.zeros((100, 100), dtype=np.float64)

    # Create a blob at row=20, col=75
    rr, cc = disk((20, 75), 8)
    img[rr, cc] = 1.0

    # Smooth it
    from scipy.ndimage import gaussian_filter

    img = gaussian_filter(img, sigma=2)

    # Find blobs
    blobs = blob_dog(img, min_sigma=3, max_sigma=15, threshold=0.1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap='gray')

    for blob in blobs:
        r, c, sigma = blob
        ax.add_patch(
            Circle((c, r), sigma * np.sqrt(2), fill=False, color='red', linewidth=2)
        )
        ax.annotate(
            f'({r:.0f}, {c:.0f})',
            (c, r),
            xytext=(c - 25, r + 25),
            color='red',
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='red'),
        )

    ax.set_title(
        'blob_dog returns (r, c, sigma)\n'
        'Blob placed at img[20, 75]\n'
        'IJ convention: returns (row~20, col~75) -> TOP-RIGHT'
    )
    ax.set_xlabel('Column (horizontal axis)')
    ax.set_ylabel('Row (vertical axis)')
    plt.tight_layout()
    plt.savefig('test_05_blob_dog.png', dpi=100)
    plt.close()
    print("Saved: test_05_blob_dog.png")


# =============================================================================
# TEST 6: skimage.measure.find_contours - demonstrates IJ convention
# =============================================================================
def test_measure_find_contours():
    """
    find_contours returns (row, col) coordinate arrays.

    We create a shape in the top-right region.
    """
    from skimage.measure import find_contours
    from skimage.draw import disk

    img = np.zeros((100, 100), dtype=np.float64)

    # Create a filled disk at row=25, col=70
    rr, cc = disk((25, 70), 15)
    img[rr, cc] = 1.0

    # Find contours
    contours = find_contours(img, 0.5)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap='gray')

    for contour in contours:
        # contour is (N, 2) array of (row, col) pairs
        # For plotting: x=col, y=row
        ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)

        # Show a sample point
        idx = 0
        r, c = contour[idx]
        ax.plot(c, r, 'go', markersize=8)
        ax.annotate(
            f'Point: ({r:.1f}, {c:.1f})',
            (c, r),
            xytext=(c - 30, r + 20),
            color='green',
            fontsize=9,
            arrowprops=dict(arrowstyle='->', color='green'),
        )

    ax.set_title(
        'find_contours returns (row, col) arrays\n'
        'Shape at img[25, 70] (TOP-RIGHT)\n'
        'IJ convention: contour coords have row~25, col~70'
    )
    ax.set_xlabel('Column (horizontal axis)')
    ax.set_ylabel('Row (vertical axis)')
    plt.tight_layout()
    plt.savefig('test_06_find_contours.png', dpi=100)
    plt.close()
    print("Saved: test_06_find_contours.png")


# =============================================================================
# TEST 7: skimage.measure.regionprops centroid - demonstrates IJ convention
# =============================================================================
def test_measure_regionprops():
    """
    regionprops.centroid returns (row, col).

    We create a labeled region in the top-right.
    """
    from skimage.measure import regionprops, label
    from skimage.draw import disk

    img = np.zeros((100, 100), dtype=np.uint8)

    # Create a region at row=20, col=75
    rr, cc = disk((20, 75), 12)
    img[rr, cc] = 1

    # Label and get properties
    labeled = label(img)
    props = regionprops(labeled)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap='gray')

    for prop in props:
        centroid = prop.centroid  # (row, col)
        r, c = centroid
        ax.plot(c, r, 'r+', markersize=20, markeredgewidth=3)
        ax.annotate(
            f'centroid: ({r:.1f}, {c:.1f})',
            (c, r),
            xytext=(c - 35, r + 25),
            color='red',
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='red'),
        )

    ax.set_title(
        'regionprops.centroid returns (row, col)\n'
        'Region at img[20, 75] (TOP-RIGHT)\n'
        'IJ convention: centroid ~ (20, 75)'
    )
    ax.set_xlabel('Column (horizontal axis)')
    ax.set_ylabel('Row (vertical axis)')
    plt.tight_layout()
    plt.savefig('test_07_regionprops.png', dpi=100)
    plt.close()
    print("Saved: test_07_regionprops.png")


# =============================================================================
# TEST 8: skimage.transform.AffineTransform - demonstrates XY convention
# =============================================================================
def test_transform_affine():
    """
    AffineTransform uses (x, y) coordinate pairs.

    We define source points and apply a transform, showing that
    the first coordinate is horizontal (x), second is vertical (y).
    """
    from skimage.transform import AffineTransform

    # Define points as (x, y) pairs
    # Point at x=80, y=20 (right side, near top in image coords)
    points = np.array(
        [
            [10, 10],  # (x=10, y=10) - near origin
            [80, 20],  # (x=80, y=20) - far right, near top
            [50, 50],  # (x=50, y=50) - center
        ],
        dtype=np.float64,
    )

    # Create a simple translation transform: shift x by 5, y by 10
    tform = AffineTransform(translation=(5, 10))

    # Apply transform
    transformed = tform(points)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a reference image
    img = np.zeros((100, 100), dtype=np.uint8)
    ax.imshow(img, cmap='gray', extent=[0, 100, 100, 0])

    # Plot original points (blue)
    for i, (x, y) in enumerate(points):
        ax.plot(x, y, 'bo', markersize=12)
        ax.annotate(
            f'src[{i}]=({x:.0f},{y:.0f})',
            (x, y),
            xytext=(x + 3, y - 5),
            color='blue',
            fontsize=9,
        )

    # Plot transformed points (red)
    for i, (x, y) in enumerate(transformed):
        ax.plot(x, y, 'rs', markersize=12)
        ax.annotate(
            f'dst[{i}]=({x:.0f},{y:.0f})',
            (x, y),
            xytext=(x + 3, y + 10),
            color='red',
            fontsize=9,
        )

    # Draw arrows showing transformation
    for (x1, y1), (x2, y2) in zip(points, transformed):
        ax.annotate(
            '',
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
        )

    ax.set_xlim(-5, 105)
    ax.set_ylim(105, -5)
    ax.set_title(
        'AffineTransform uses (x, y) coordinates\n'
        'translation=(5, 10) shifts x by 5, y by 10\n'
        'XY convention: first coord is HORIZONTAL'
    )
    ax.set_xlabel('X (horizontal axis) - first coordinate')
    ax.set_ylabel('Y (vertical axis) - second coordinate')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('test_08_affine_transform.png', dpi=100)
    plt.close()
    print("Saved: test_08_affine_transform.png")


# =============================================================================
# TEST 9: skimage.transform.rotate center parameter - demonstrates XY convention
# =============================================================================
def test_transform_rotate_center():
    """
    rotate() center parameter uses (cols, rows) = (x, y) convention.

    This is explicitly documented as "contrary to normal skimage ordering".
    We rotate around an off-center point to show the convention.
    """
    from skimage.transform import rotate
    from skimage.draw import disk

    # Create image with a marker in top-left
    img = np.zeros((100, 100), dtype=np.float64)
    rr, cc = disk((20, 20), 10)
    img[rr, cc] = 1.0

    # Rotate 90 degrees around center=(75, 25) which means x=75, y=25
    # In image coords: col=75, row=25 (top-right area)
    center = (75, 25)  # (x, y) = (col, row)

    rotated = rotate(img, 90, center=center, resize=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original
    axes[0].imshow(img, cmap='gray')
    axes[0].plot(20, 20, 'r+', markersize=15, markeredgewidth=2)
    axes[0].plot(75, 25, 'go', markersize=10, markeredgewidth=2)
    axes[0].annotate(
        'disk at (r=20,c=20)',
        (20, 20),
        xytext=(30, 40),
        color='red',
        arrowprops=dict(arrowstyle='->', color='red'),
    )
    axes[0].annotate(
        'center=(75,25)\n=(x=75,y=25)',
        (75, 25),
        xytext=(50, 50),
        color='green',
        arrowprops=dict(arrowstyle='->', color='green'),
    )
    axes[0].set_title('Original\nDisk at top-left, rotation center at top-right')
    axes[0].set_xlabel('Column (x)')
    axes[0].set_ylabel('Row (y)')

    # Rotated
    axes[1].imshow(rotated, cmap='gray')
    axes[1].plot(75, 25, 'go', markersize=10, markeredgewidth=2)
    axes[1].annotate(
        'rotation center\n(unchanged)',
        (75, 25),
        xytext=(50, 50),
        color='green',
        arrowprops=dict(arrowstyle='->', color='green'),
    )
    axes[1].set_title(
        'After rotate(90°, center=(75, 25))\n'
        'center=(x, y) = (col, row) - XY CONVENTION\n'
        '"contrary to normal skimage ordering"'
    )
    axes[1].set_xlabel('Column (x)')
    axes[1].set_ylabel('Row (y)')

    plt.tight_layout()
    plt.savefig('test_09_rotate_center.png', dpi=100)
    plt.close()
    print("Saved: test_09_rotate_center.png")


# =============================================================================
# TEST 10: skimage.transform.hough_circle - demonstrates IJ convention
# =============================================================================
def test_hough_circle():
    """
    hough_circle: circle_perimeter takes (r, c) and the accumulator
    peak location corresponds to (row, col) of circle center.
    """
    from skimage.transform import hough_circle, hough_circle_peaks
    from skimage.draw import circle_perimeter

    img = np.zeros((100, 100), dtype=np.uint8)

    # Draw circle with center at row=25, col=70, radius=20
    rr, cc = circle_perimeter(25, 70, 20)
    valid = (rr >= 0) & (rr < 100) & (cc >= 0) & (cc < 100)
    img[rr[valid], cc[valid]] = 1

    # Detect circles
    hough_radii = np.arange(15, 25)
    hough_res = hough_circle(img, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(
        hough_res, hough_radii, total_num_peaks=1
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original with drawn circle
    axes[0].imshow(img, cmap='gray')
    axes[0].plot(70, 25, 'r+', markersize=15, markeredgewidth=2)
    axes[0].annotate(
        'Drawn at (r=25, c=70)',
        (70, 25),
        xytext=(40, 50),
        color='red',
        arrowprops=dict(arrowstyle='->', color='red'),
    )
    axes[0].set_title(
        'circle_perimeter(r=25, c=70, radius=20)\n' 'IJ convention: center at TOP-RIGHT'
    )
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')

    # Detection result
    axes[1].imshow(img, cmap='gray')
    if len(cx) > 0:
        # Note: hough_circle_peaks returns (cx, cy) which are actually (col, row)
        # despite the misleading variable names
        from matplotlib.patches import Circle

        c, r = cx[0], cy[0]
        axes[1].add_patch(
            Circle((c, r), radii[0], fill=False, color='red', linewidth=2)
        )
        axes[1].plot(c, r, 'r+', markersize=15, markeredgewidth=2)
        axes[1].annotate(
            f'Detected: cx={c}, cy={r}, r={radii[0]}',
            (c, r),
            xytext=(20, 60),
            color='red',
            arrowprops=dict(arrowstyle='->', color='red'),
        )

    axes[1].set_title(
        'hough_circle_peaks returns (cx, cy)\n'
        f'Found center at cx={cx[0] if len(cx) else "?"}, cy={cy[0] if len(cy) else "?"}\n'
        'Note: cx=column, cy=row (IJ convention)'
    )
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')

    plt.tight_layout()
    plt.savefig('test_10_hough_circle.png', dpi=100)
    plt.close()
    print("Saved: test_10_hough_circle.png")


# =============================================================================
# TEST 11: skimage.transform.hough_ellipse - demonstrates MIXED convention
# =============================================================================
def test_hough_ellipse():
    """
    hough_ellipse has MIXED conventions:
    - Returns (accumulator, yc, xc, a, b, orientation) where yc comes before xc
    - The docstring example uses img[cc, rr] (swapped!)

    This test demonstrates the confusion.
    """
    from skimage.draw import ellipse_perimeter

    # Draw ellipse at center (row=25, col=55), with semi-axes
    # ellipse_perimeter(r, c, r_radius, c_radius)
    rr, cc = ellipse_perimeter(25, 55, 12, 18)
    valid = (rr >= 0) & (rr < 80) & (cc >= 0) & (cc < 80)

    # Following the DOCSTRING example: img[cc, rr] = 1 (SWAPPED!)
    # This is the documented usage but seems wrong
    img_docstring = np.zeros((80, 80), dtype=np.uint8)
    img_docstring[cc[valid], rr[valid]] = 1

    # What we'd expect with IJ convention: img[rr, cc] = 1
    img_expected = np.zeros((80, 80), dtype=np.uint8)
    img_expected[rr[valid], cc[valid]] = 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # As per docstring (swapped)
    axes[0].imshow(img_docstring, cmap='gray')
    axes[0].set_title(
        'Following hough_ellipse docstring:\n'
        'ellipse_perimeter(25, 55, 12, 18)\n'
        'then img[cc, rr] = 1 (SWAPPED!)\n'
        'Ellipse appears at WRONG location'
    )
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    axes[0].plot(
        55, 25, 'r+', markersize=15, markeredgewidth=2, label='Expected center (55, 25)'
    )
    axes[0].legend()

    # Expected IJ convention
    axes[1].imshow(img_expected, cmap='gray')
    axes[1].set_title(
        'Expected with IJ convention:\n'
        'ellipse_perimeter(25, 55, 12, 18)\n'
        'then img[rr, cc] = 1 (CORRECT)\n'
        'Ellipse at TOP-RIGHT as expected'
    )
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    axes[1].plot(
        55, 25, 'r+', markersize=15, markeredgewidth=2, label='Center at (r=25, c=55)'
    )
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('test_11_hough_ellipse_bug.png', dpi=100)
    plt.close()
    print("Saved: test_11_hough_ellipse_bug.png")


# =============================================================================
# TEST 12: skimage.filters.rank shift_x, shift_y - demonstrates XY convention
# =============================================================================
def test_rank_filter_shifts():
    """
    rank filters use shift_x and shift_y to offset the footprint center.

    In core_cy.pyx:
        centre_r = footprint.shape[0] / 2 + shift_y  # shift_y affects ROW
        centre_c = footprint.shape[1] / 2 + shift_x  # shift_x affects COLUMN

    This is XY convention: x=horizontal (column), y=vertical (row).

    We demonstrate by applying a maximum filter with an asymmetric shift
    and showing that shift_x moves the effect horizontally (in columns)
    while shift_y moves it vertically (in rows).
    """
    from skimage.filters import rank
    from skimage.morphology import footprint_rectangle

    # Create an image with a single bright pixel
    img = np.zeros((50, 80), dtype=np.uint8)
    img[25, 40] = 255  # Bright pixel at row=25, col=40

    # Small footprint
    footprint = footprint_rectangle((5, 5))

    # No shift - filter centered
    result_no_shift = rank.maximum(img, footprint, shift_x=0, shift_y=0)

    # shift_x=+2: shifts footprint center RIGHT (in column direction)
    # This means the bright pixel will affect pixels to its LEFT
    result_shift_x = rank.maximum(img, footprint, shift_x=2, shift_y=0)

    # shift_y=+2: shifts footprint center DOWN (in row direction)
    # This means the bright pixel will affect pixels ABOVE it
    result_shift_y = rank.maximum(img, footprint, shift_x=0, shift_y=2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].plot(40, 25, 'r+', markersize=15, markeredgewidth=2)
    axes[0, 0].set_title('Original: bright pixel at (row=25, col=40)')
    axes[0, 0].set_xlabel('Column')
    axes[0, 0].set_ylabel('Row')

    # No shift
    axes[0, 1].imshow(result_no_shift, cmap='gray')
    axes[0, 1].plot(40, 25, 'r+', markersize=15, markeredgewidth=2)
    axes[0, 1].set_title(
        'rank.maximum with shift_x=0, shift_y=0\n'
        'Footprint centered - effect symmetric around pixel'
    )
    axes[0, 1].set_xlabel('Column')
    axes[0, 1].set_ylabel('Row')

    # shift_x only
    axes[1, 0].imshow(result_shift_x, cmap='gray')
    axes[1, 0].plot(40, 25, 'r+', markersize=15, markeredgewidth=2)
    axes[1, 0].annotate(
        'shift_x=+2 moves\neffect LEFT',
        (35, 25),
        xytext=(15, 15),
        color='red',
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='red'),
    )
    axes[1, 0].set_title(
        'rank.maximum with shift_x=+2, shift_y=0\n'
        'XY convention: shift_x affects COLUMNS (horizontal)\n'
        'Positive shift_x moves footprint center RIGHT'
    )
    axes[1, 0].set_xlabel('Column (shift_x direction)')
    axes[1, 0].set_ylabel('Row')

    # shift_y only
    axes[1, 1].imshow(result_shift_y, cmap='gray')
    axes[1, 1].plot(40, 25, 'r+', markersize=15, markeredgewidth=2)
    axes[1, 1].annotate(
        'shift_y=+2 moves\neffect UP',
        (40, 20),
        xytext=(50, 10),
        color='red',
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='red'),
    )
    axes[1, 1].set_title(
        'rank.maximum with shift_x=0, shift_y=+2\n'
        'XY convention: shift_y affects ROWS (vertical)\n'
        'Positive shift_y moves footprint center DOWN'
    )
    axes[1, 1].set_xlabel('Column')
    axes[1, 1].set_ylabel('Row (shift_y direction)')

    plt.suptitle(
        'skimage.filters.rank: shift_x and shift_y use XY convention\n'
        'shift_x = column shift (j), shift_y = row shift (i)',
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig('test_12_rank_filter_shifts.png', dpi=100)
    plt.close()
    print("Saved: test_12_rank_filter_shifts.png")


# =============================================================================
# TEST 13: Comparison showing IJ vs XY difference
# =============================================================================
def test_convention_comparison():
    """
    Side-by-side comparison showing what happens when you interpret
    coordinates as IJ vs XY.
    """
    from skimage.draw import disk

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Same coordinate pair: (20, 70)
    coord = (20, 70)

    # IJ interpretation: row=20, col=70 -> top-right
    img_ij = np.zeros((100, 100), dtype=np.uint8)
    rr, cc = disk(coord, 12)
    img_ij[rr, cc] = 255

    axes[0].imshow(img_ij, cmap='gray')
    axes[0].set_title(
        f'IJ Convention: coord ({coord[0]}, {coord[1]})\n'
        f'row={coord[0]}, col={coord[1]}\n'
        'Disk appears at TOP-RIGHT'
    )
    axes[0].set_xlabel('Column (second coordinate)')
    axes[0].set_ylabel('Row (first coordinate)')
    axes[0].plot(coord[1], coord[0], 'r+', markersize=20, markeredgewidth=3)

    # XY interpretation: x=20, y=70 -> bottom-left
    img_xy = np.zeros((100, 100), dtype=np.uint8)
    # For XY: x=20 means col=20, y=70 means row=70
    rr_xy, cc_xy = disk((coord[1], coord[0]), 12)  # swap to simulate XY interpretation
    img_xy[rr_xy, cc_xy] = 255

    axes[1].imshow(img_xy, cmap='gray')
    axes[1].set_title(
        f'XY Convention: coord ({coord[0]}, {coord[1]})\n'
        f'x={coord[0]}, y={coord[1]}\n'
        'Disk appears at BOTTOM-LEFT'
    )
    axes[1].set_xlabel('X (first coordinate)')
    axes[1].set_ylabel('Y (second coordinate)')
    axes[1].plot(coord[0], coord[1], 'r+', markersize=20, markeredgewidth=3)

    plt.suptitle('Same coordinates (20, 70) interpreted differently', fontsize=14)
    plt.tight_layout()
    plt.savefig('test_13_convention_comparison.png', dpi=100)
    plt.close()
    print("Saved: test_13_convention_comparison.png")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("Running coordinate convention tests...\n")

    test_draw_line()
    test_draw_disk()
    test_draw_ellipse()
    test_feature_peak_local_max()
    test_feature_blob_dog()
    test_measure_find_contours()
    test_measure_regionprops()
    test_transform_affine()
    test_transform_rotate_center()
    test_hough_circle()
    test_hough_ellipse()
    test_rank_filter_shifts()
    test_convention_comparison()

    print("\nAll tests complete. Check the generated PNG files.")
