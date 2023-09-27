import taichi as ti
from taichi.math import mat3, vec3, pi, sqrt, sin, cos, mat4, vec4, length, dot, exp

vecD = ti.types.vector(32, dtype=ti.f32)

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
    - density: extinction coeff
    """
    density: float


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
    S_mat: mat3
    density: float
    albedo: vec3
    
    
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

@ti.func
def inverse_cdf(weights, nbins):
    """
    Inverse sampling a PDF. Assume that weights sum up to 1 and the bins 
    distribute between [0, 1].

    Return a value between 0 and 1.
    """
    cdf = vecD(0.0)
    below = 0
    above = 0
    u = ti.random()

    for i in range(nbins):
        cdf[i+1] = cdf[i] + weights[i]
        if u > cdf[i+1]:
            below = i
            above = i+1
            break
    
    return lerp(u, cdf[below], cdf[above], below/nbins, above/nbins)


@ti.kernel
def to_image(img: ti.template(), out: ti.template(), spp: int):
    for I in ti.grouped(img):
        img[I] = out[I] / spp

@ti.func
def sd_bunny(p: vec3) -> float:
    # sdf is undefined outside the unit sphere, uncomment to witness the abominations
    sd = 0.0
    if length(p) > 1.0:
        sd =  length(p) - 0.8
    else:
        # neural networks can be really compact... when they want to be
        f00=sin(p.y*vec4(-3.02,1.95,-3.42,-.60)+p.z*vec4(3.08,.85,-2.25,-.24)-p.x*vec4(-.29,1.16,-3.74,2.89)+vec4(-.71,4.50,-3.24,-3.50))
        f01=sin(p.y*vec4(-.40,-3.61,3.23,-.14)+p.z*vec4(-.36,3.64,-3.91,2.66)-p.x*vec4(2.90,-.54,-2.75,2.71)+vec4(7.02,-5.41,-1.12,-7.41))
        f02=sin(p.y*vec4(-1.77,-1.28,-4.29,-3.20)+p.z*vec4(-3.49,-2.81,-.64,2.79)-p.x*vec4(3.15,2.14,-3.85,1.83)+vec4(-2.07,4.49,5.33,-2.17))
        f03=sin(p.y*vec4(-.49,.68,3.05,.42)+p.z*vec4(-2.87,.78,3.78,-3.41)-p.x*vec4(-2.65,.33,.07,-.64)+vec4(-3.24,-5.90,1.14,-4.71))
        f10=sin(f00@mat4(-.34,.06,-.59,-.76,.10,-.19,-.12,.44,.64,-.02,-.26,.15,-.16,.21,.91,.15)+
            f01@mat4(.01,.54,-.77,.11,.06,-.14,.43,.51,-.18,.08,.39,.20,.33,-.49,-.10,.19)+
            f02@mat4(.27,.22,.43,.53,.18,-.17,.23,-.64,-.14,.02,-.10,.16,-.13,-.06,-.04,-.36)+
            f03@mat4(-.13,.29,-.29,.08,1.13,.02,-.83,.32,-.32,.04,-.31,-.16,.14,-.03,-.20,.39)+
            vec4(.73,-4.28,-1.56,-1.80))+f00
        f11=sin(f00@mat4(-1.11,.55,-.12,-1.00,.16,.15,-.30,.31,-.01,.01,.31,-.42,-.29,.38,-.04,.71)+
            f01@mat4(.96,-.02,.86,.52,-.14,.60,.44,.43,.02,-.15,-.49,-.05,-.06,-.25,-.03,-.22)+
            f02@mat4(.52,.44,-.05,-.11,-.56,-.10,-.61,-.40,-.04,.55,.32,-.07,-.02,.28,.26,-.49)+
            f03@mat4(.02,-.32,.06,-.17,-.59,.00,-.24,.60,-.06,.13,-.21,-.27,-.12,-.14,.58,-.55)+
            vec4(-2.24,-3.48,-.80,1.41))+f01
        f12=sin(f00@mat4(.44,-.06,-.79,-.46,.05,-.60,.30,.36,.35,.12,.02,.12,.40,-.26,.63,-.21)+
            f01@mat4(-.48,.43,-.73,-.40,.11,-.01,.71,.05,-.25,.25,-.28,-.20,.32,-.02,-.84,.16)+
            f02@mat4(.39,-.07,.90,.36,-.38,-.27,-1.86,-.39,.48,-.20,-.05,.10,-.00,-.21,.29,.63)+
            f03@mat4(.46,-.32,.06,.09,.72,-.47,.81,.78,.90,.02,-.21,.08,-.16,.22,.32,-.13)+
            vec4(3.38,1.20,.84,1.41))+f02
        f13=sin(f00@mat4(-.41,-.24,-.71,-.25,-.24,-.75,-.09,.02,-.27,-.42,.02,.03,-.01,.51,-.12,-1.24)+
            f01@mat4(.64,.31,-1.36,.61,-.34,.11,.14,.79,.22,-.16,-.29,-.70,.02,-.37,.49,.39)+
            f02@mat4(.79,.47,.54,-.47,-1.13,-.35,-1.03,-.22,-.67,-.26,.10,.21,-.07,-.73,-.11,.72)+
            f03@mat4(.43,-.23,.13,.09,1.38,-.63,1.57,-.20,.39,-.14,.42,.13,-.57,-.08,-.21,.21)+
            vec4(-.34,-3.28,.43,-.52))+f03
        f00=sin(f10@mat4(-.72,.23,-.89,.52,.38,.19,-.16,-.88,.26,-.37,.09,.63,.29,-.72,.30,-.95)+
            f11@mat4(-.22,-.51,-.42,-.73,-.32,.00,-1.03,1.17,-.20,-.03,-.13,-.16,-.41,.09,.36,-.84)+
            f12@mat4(-.21,.01,.33,.47,.05,.20,-.44,-1.04,.13,.12,-.13,.31,.01,-.34,.41,-.34)+
            f13@mat4(-.13,-.06,-.39,-.22,.48,.25,.24,-.97,-.34,.14,.42,-.00,-.44,.05,.09,-.95)+
            vec4(.48,.87,-.87,-2.06))/1.4+f10
        f01=sin(f10@mat4(-.27,.29,-.21,.15,.34,-.23,.85,-.09,-1.15,-.24,-.05,-.25,-.12,-.73,-.17,-.37)+
            f11@mat4(-1.11,.35,-.93,-.06,-.79,-.03,-.46,-.37,.60,-.37,-.14,.45,-.03,-.21,.02,.59)+
            f12@mat4(-.92,-.17,-.58,-.18,.58,.60,.83,-1.04,-.80,-.16,.23,-.11,.08,.16,.76,.61)+
            f13@mat4(.29,.45,.30,.39,-.91,.66,-.35,-.35,.21,.16,-.54,-.63,1.10,-.38,.20,.15)+
            vec4(-1.72,-.14,1.92,2.08))/1.4+f11
        f02=sin(f10@mat4(1.00,.66,1.30,-.51,.88,.25,-.67,.03,-.68,-.08,-.12,-.14,.46,1.15,.38,-.10)+
            f11@mat4(.51,-.57,.41,-.09,.68,-.50,-.04,-1.01,.20,.44,-.60,.46,-.09,-.37,-1.30,.04)+
            f12@mat4(.14,.29,-.45,-.06,-.65,.33,-.37,-.95,.71,-.07,1.00,-.60,-1.68,-.20,-.00,-.70)+
            f13@mat4(-.31,.69,.56,.13,.95,.36,.56,.59,-.63,.52,-.30,.17,1.23,.72,.95,.75)+
            vec4(-.90,-3.26,-.44,-3.11))/1.4+f12
        f03=sin(f10@mat4(.51,-.98,-.28,.16,-.22,-.17,-1.03,.22,.70,-.15,.12,.43,.78,.67,-.85,-.25)+
            f11@mat4(.81,.60,-.89,.61,-1.03,-.33,.60,-.11,-.06,.01,-.02,-.44,.73,.69,1.02,.62)+
            f12@mat4(-.10,.52,.80,-.65,.40,-.75,.47,1.56,.03,.05,.08,.31,-.03,.22,-1.63,.07)+
            f13@mat4(-.18,-.07,-1.22,.48,-.01,.56,.07,.15,.24,.25,-.09,-.54,.23,-.08,.20,.36)+
            vec4(-1.11,-4.28,1.02,-.23))/1.4+f13
        sd = dot(f00,vec4(.09,.12,-.07,-.03))+dot(f01,vec4(-.04,.07,-.08,.05))+dot(f02,vec4(-.01,.06,-.02,.07))+dot(f03,vec4(-.05,.07,.03,.04))-0.16

    return sd

@ti.func
def laplacian_cdf(s: float, beta=10):
    if s <= 0:
        return 0.5 * exp(s/beta)
    else:
        return 1 - 0.5 * exp(-s/beta)
    pass