import numpy as np
import skimage.transform as st1
import _skimage2.transform as st2
import skimage.data as data

def test_warp_invariance():
    img = data.camera().astype(float)
    
    # 1. Translation Test
    # skimage (x, y): shift right 20, down 10
    t1 = st1.AffineTransform(translation=(20, 10))
    w1 = st1.warp(img, t1.inverse)
    
    # _skimage2 (i, j): shift down 10, right 20
    t2 = st2.AffineTransform(translation=(10, 20))
    w2 = st2.warp(img, t2.inverse)
    
    diff = np.abs(w1 - w2).max()
    print(f"Translation warp diff: {diff}")
    assert np.allclose(w1, w2)

def test_rotate_invariance():
    img = data.camera().astype(float)
    angle = 30
    
    # skimage rotate
    r1 = st1.rotate(img, angle)
    
    # _skimage2 rotate
    r2 = st2.rotate(img, angle)
    
    diff = np.abs(r1 - r2).max()
    print(f"Rotation warp diff: {diff}")
    assert np.allclose(r1, r2)

def test_similarity_rotation():
    # Test that rotation angle is interpreted same relative to physical structure
    # In 2D, rotation clockwise by theta.
    # For (x, y), x is horizontal, y is vertical.
    # For (i, j), i is vertical, j is horizontal.
    
    angle = np.deg2rad(30)
    
    # skimage: rotation 30 deg clockwise around (0,0)
    t1 = st1.SimilarityTransform(rotation=angle)
    # _skimage2: rotation 30 deg clockwise around (0,0)
    t2 = st2.SimilarityTransform(rotation=angle)
    
    # If we transform a point on the "first" axis:
    p1_src = np.array([[10, 0]]) # x=10, y=0
    p1_dst = t1(p1_src)
    # x' = 10*cos(30), y' = 10*sin(30)
    
    p2_src = np.array([[10, 0]]) # i=10, j=0
    p2_dst = t2(p2_src)
    # i' = 10*cos(30), j' = 10*sin(30)
    
    print(f"p1_dst: {p1_dst}")
    print(f"p2_dst: {p2_dst}")
    
    assert np.allclose(p1_dst, p2_dst)

if __name__ == "__main__":
    test_warp_invariance()
    test_rotate_invariance()
    test_similarity_rotation()
    print("Tests passed!")
