import taichi as ti
import numpy as np
from taichi.math import pi, vec3, vec4, mat3, clamp, sqrt, dot, inverse, determinant

from volume import AABB
from utils import trilerp

@ti.dataclass
class SGGXVoxel:
    albedo: vec3
    density: float
    S_mat: mat3


@ti.data_oriented
class SGGX:

    def __init__(self, v2w, res, lerp) -> None:
        self.aabb = AABB(lt=vec3(0,0,0), rt=vec3(1,1,1)) # In local space, aabb assumed to be [0,0,0] and [1,1,1]
        self.v2w = v2w
        self.res = res
        self.lerp = lerp
        self.data = SGGXVoxel.field(shape=(res, res, res))

    @ti.func
    def at(self, x: vec3) -> SGGXVoxel:
        tmp = vec4(x[0], x[1], x[2], 1)
        tmp = self.w2v @ tmp
        x = vec3(tmp.x, tmp.y, tmp.z)
        x = clamp(x, xmin=vec3([0,0,0]), xmax=vec3([0.999,0.999,0.999]))
        x = x * (self.res - 1)
        x0 = ti.cast(x, int)
        x1 = x0 + 1
        if self.lerp:
            albedo = trilerp(x, x0, x1, 
                        self.data.albedo[x0[0], x0[1], x0[2]],
                        self.data.albedo[x1[0], x0[1], x0[2]],
                        self.data.albedo[x0[0], x1[1], x0[2]],
                        self.data.albedo[x1[0], x1[1], x0[2]],
                        self.data.albedo[x0[0], x0[1], x1[2]],
                        self.data.albedo[x1[0], x0[1], x1[2]],
                        self.data.albedo[x0[0], x1[1], x1[2]],
                        self.data.albedo[x1[0], x1[1], x1[2]])
            
            density = trilerp(x, x0, x1, 
                        self.data.density[x0[0], x0[1], x0[2]],
                        self.data.density[x1[0], x0[1], x0[2]],
                        self.data.density[x0[0], x1[1], x0[2]],
                        self.data.density[x1[0], x1[1], x0[2]],
                        self.data.density[x0[0], x0[1], x1[2]],
                        self.data.density[x1[0], x0[1], x1[2]],
                        self.data.density[x0[0], x1[1], x1[2]],
                        self.data.density[x1[0], x1[1], x1[2]])
            
            S_mat = trilerp(x, x0, x1, 
                        self.data.eigen[x0[0], x0[1], x0[2]],
                        self.data.eigen[x1[0], x0[1], x0[2]],
                        self.data.eigen[x0[0], x1[1], x0[2]],
                        self.data.eigen[x1[0], x1[1], x0[2]],
                        self.data.eigen[x0[0], x0[1], x1[2]],
                        self.data.eigen[x1[0], x0[1], x1[2]],
                        self.data.eigen[x0[0], x1[1], x1[2]],
                        self.data.eigen[x1[0], x1[1], x1[2]])
            voxel = SGGXVoxel(albedo, density, S_mat)
        else:
            voxel = self.data[x0]
        
        return voxel

    @ti.func
    def Tr(self, x: vec3, y: vec3):

        pass

    @ti.func
    def specular_phase_func(self, x: vec3, wi: vec3, wo: vec3):
        wh = (wi + wo) / (wi + wo).norm()
        voxel = self.at(x)
        S = voxel.S_mat
        return 0.25 * 1 / (pi * sqrt(determinant(S)) * (dot(wh, inverse(S) @ wh))**2) / sqrt(dot(wi, S @ wi))

    @ti.func
    def diffuse_phase_func(self):
        pass

    @ti.func
    def sample_specular(self):
        pass

    @ti.func
    def sample_diffuse(self):
        pass

    @ti.func
    def pdf_specular(self):
        pass

    @ti.func
    def pdf_diffuse(self):
        pass

    @ti.func
    def ndf(self, x: vec3, m: vec3):
        """
        Normal distribution function
        """
        voxel = self.at(x)
        S = voxel.S_mat
        return 1 / (pi * sqrt(determinant(S)) * (dot(m, inverse(S) @ m))**2)

    @ti.func
    def sigma(self, x: vec3, w: vec3):
        voxel = self.at(x)
        S = voxel.S_mat
        return sqrt(dot(w, S @ w))

    @ti.func
    def scatter_coeff_dir(self, x: vec3, w: vec3):
        voxel = self.at(x)
        S = voxel.S_mat
        density = voxel.density
        albedo = voxel.albedo
        return albedo * density * sqrt(dot(w, S @ w))

    @ti.func
    def extinct_coeff_dir(self, x: vec3, w: vec3):
        voxel = self.at(x)
        S = voxel.S_mat
        density = voxel.density
        albedo = voxel.albedo
        return albedo * density * sqrt(dot(w, S @ w))