# [nero camfront 2026-11-09] camera0 geometry: the rectification maps send each pinhole pixel to the fisheye pixel of the same ray.
import math
import numpy as np
from autoware_nero_camera0.camera0_node import kb_project, rect_maps

F, CX, CY, K = 921.73, 961.56, 772.15, [0.00563, -0.01202, -0.0196, 0.0]


def test_centre_maps_to_principal_point():
    m1, m2, cxr, cyr = rect_maps(F, F, CX, CY, K, 1440, 1080, 604.15)
    import cv2
    mx, my = cv2.convertMaps(m1, m2, cv2.CV_32FC1)
    # the rectified centre (719.5, 539.5) lies between four pixels; CV_16SC2 maps are quantised to 1/32 px
    assert abs(mx[539:541, 719:721].mean() - CX) < 0.1 and abs(my[539:541, 719:721].mean() - CY) < 0.1


def test_rays_agree():
    f_r = (1440 / 2) / math.tan(math.radians(50))
    m1, m2, cxr, cyr = rect_maps(F, F, CX, CY, K, 1440, 1080, f_r)
    import cv2
    mx, my = cv2.convertMaps(m1, m2, cv2.CV_32FC1)
    for u, v in ((0, 0), (1439, 1079), (100, 900), (1300, 200)):
        ray = np.array([(u - cxr) / f_r, (v - cyr) / f_r, 1.0])
        x, y = kb_project(ray[None, :], F, F, CX, CY, K)
        assert abs(x[0] - mx[v, u]) < 0.6 and abs(y[0] - my[v, u]) < 0.6


def test_equidistant_without_k():
    th = math.radians(40)
    x, y = kb_project(np.array([[math.sin(th), 0.0, math.cos(th)]]), F, F, CX, CY, [0, 0, 0, 0])
    assert abs((x[0] - CX) - F * th) < 1e-6
