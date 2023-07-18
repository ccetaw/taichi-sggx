import taichi as ti
from taichi.math import mat3, vec3, pi, sqrt, sin, cos

@ti.func
def lerp(x, x0, x1, q0, q1):
    fx = (x1 - x)/(x1 - x0)
    return fx*q0 + (1-fx)*q1

@ti.func
def bilerp(x, x0, x1, q00, q10, q01, q11):
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
def build_orthonomal_basis(w3: vec3) -> mat3:
    w1 = vec3(0)
    w2 = vec3(0)
    if (w3.z < -0.9999999):
        pass