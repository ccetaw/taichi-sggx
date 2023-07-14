import taichi as ti
from taichi.math import vec3

@ti.func
def lerp(x, x0, x1, q0, q1):
    a = (x1 - x)/(x1 - x0)
    return a*q0 + (1-a)*q1

@ti.func
def bilerp(x, x0, x1, q00, q10, q01, q11):
    qx0 = lerp(x[0], x0[0], x1[0], q00, q10)
    qx1 = lerp(x[0], x0[0], x1[0], q01, q11)
    return lerp(x[1], x0[1], x1[1], qx0, qx1)

@ti.func
def trilerp(x, x0, x1, q000, q100, q010, q110, q001, q101, q011, q111):
    qx00 = lerp(x[0], x0[0], x1[0], q000, q100)
    qx10 = lerp(x[0], x0[0], x1[0], q010, q110)
    qx01 = lerp(x[0], x0[0], x1[0], q001, q101)
    qx11 = lerp(x[0], x0[0], x1[0], q011, q111)

    qxy0 = lerp(x[1], x0[1], x1[1], qx00, qx10)
    qxy1 = lerp(x[1], x0[1], x1[1], qx01, qx11)

    return lerp(x[2], x0[2], x1[2], qxy0, qxy1)

@ti.func 
def sample_uniform_sphere():
    u = ti.random()
    v = ti.random()

    return vec3()