import taichi as ti
from taichi.math import mat3, vec3, pi, sqrt, sin, cos, exp, distance

@ti.dataclass
class Ray:
    """
    Ray dataclass
    -------------
    
    Members:
    - o: origin
    - d: direction
    """
    o: vec3
    d: vec3

@ti.dataclass
class AABB:
    """
    Axis aligned bounding box dataclass.
    -------

    Members:
    - lb: left-bottom
    - rt: right-top
    """
    lb: vec3
    rt: vec3

@ti.dataclass
class Voxel:
    """
    Voxel dataclass.
    ------

    Members:
    - density: 3-channel extinction coeff
    - albedo: 3-channel color
    """
    density: vec3
    albedo: vec3

@ti.dataclass
class SGGXVoxel:
    """
    SGGX voxel dataclass
    --------

    Members:
    - density: single channel extinction coeff
    - albedol: 3-channel color
    - S_mat: 3x3 SGGX matrix
    """
    albedo: vec3
    density: float
    S_mat: mat3


@ti.func
def lerp(x, x0, x1, q0, q1):
    """
    Linear interpolation. 

    Input: 
    - x: interpolate at x
    - x0, x1: 1D points 
    - q0, q1: values at x0, x1

    Output:
    - value at x
    """
    fx = (x1 - x)/(x1 - x0)
    return fx*q0 + (1-fx)*q1

@ti.func
def bilerp(x, x0, x1, q00, q10, q01, q11):
    """
    Bilinear interpolation.

    Input:
    - x: interpolate at x
    - x0, x1: 2D points, bottom-left and top-right corner
    - q00, q10, q01, q11: values at bottom-left, bottom-right, top-left, top-right

    Output:
    - value at x
    """

    fx = (x1[0] - x[0]) / (x1[0] - x0[0])
    fy = (x1[1] - x[1]) / (x1[1] - x0[1])
    _fx = 1 - fx
    _fy = 1 - fy
    return (q00 * fx + q10 * _fx) * fy + (q01 * fx + q11 * _fx) * _fy
    # qx0 = lerp(x[0], x0[0], x1[0], q00, q10)
    # qx1 = lerp(x[0], x0[0], x1[0], q01, q11)
    # return lerp(x[1], x0[1], x1[1], qx0, qx1)

@ti.func
def trilerp(x, x0, x1, q000, q100, q010, q110, q001, q101, q011, q111):
    """
    Trilinear interpolation.

    Input:
    - x: interpolate at x
    - x0, x1: 3D points in a right-handed system. In unit length, x0=(0,0,0), x1=(1,1,1)
    - q000, q010, ...: values at (0,0,0), (0,1,0), ...

    Output:
    - value at x
    """
    fx = (x1[0] - x[0]) / (x1[0] - x0[0])
    fy = (x1[1] - x[1]) / (x1[1] - x0[1])
    fz = (x1[2] - x[2]) / (x1[2] - x0[2])
    _fx = 1 - fx
    _fy = 1 - fy
    _fz = 1 - fz
    return ((q000 * fx + q100 * _fx) * fy + (q010 * fx + q110 * _fx) * _fy) * fz + ((q001 * fx + q101 * _fx) * fy + (q011 * fx + q111 * _fx) * _fy) * _fz
    # qx00 = lerp(x[0], x0[0], x1[0], q000, q100)
    # qx10 = lerp(x[0], x0[0], x1[0], q010, q110)
    # qx01 = lerp(x[0], x0[0], x1[0], q001, q101)
    # qx11 = lerp(x[0], x0[0], x1[0], q011, q111)

    # qxy0 = lerp(x[1], x0[1], x1[1], qx00, qx10)
    # qxy1 = lerp(x[1], x0[1], x1[1], qx01, qx11)

    # return lerp(x[2], x0[2], x1[2], qxy0, qxy1)

@ti.func 
def sample_uniform_sphere() -> vec3:
    u = ti.random()
    v = ti.random()

    z = 2 * u - 1
    phi = 2 * pi * v
    r = sqrt(1 - z*z)
    x = r * cos(phi)
    y = r * sin(phi)

    return vec3(x, y, z)

@ti.func
def build_orthonomal_basis(w3: vec3) -> tuple[vec3, vec3]:
    """
    Building an Orthonormal Basis from a 3D Unit Vector Without Normalization
    Jeppe Revall Frisvad. Building an orthonormal basis from a 3d unit vector without normalization.
    Journal of Graphics Tools, 16(3):151–159, 2012.
    """
    w1 = vec3(0)
    w2 = vec3(0)
    if (w3.z < -0.9999999):
        w1 = vec3(0.0, -1.0, 0.0)
        w2 = vec3(-1.0, 0.0, 0.0)
    else:
        a = 1.0 / (1.0 + w3.z)
        b = -w3.x * w3.y * a
        w1 = vec3(1 - w3.x * w3.x * a, b, -w3.x)
        w2 = vec3(b, 1 - w3.y * w3.y * a, -w3.y)
    
    return w1, w2