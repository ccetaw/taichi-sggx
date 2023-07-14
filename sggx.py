import taichi as ti
import numpy as np
from taichi.math import vec3, vec4, mat3, clamp

from volume import AABB
from utils import trilerp

@ti.dataclass
class SGGXVoxel:
    albedo: vec3
    density: float
    eigen: vec3
    normal: vec3


@ti.data_oriented
class SGGX:

    def __init__(self, v2w, res) -> None:
        self.aabb = AABB(lt=vec3(0,0,0), rt=vec3(1,1,1)) # In local space, aabb assumed to be [0,0,0] and [1,1,1]
        self.v2w = v2w
        self.res = res
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
        
        eigen = trilerp(x, x0, x1, 
                       self.data.eigen[x0[0], x0[1], x0[2]],
                       self.data.eigen[x1[0], x0[1], x0[2]],
                       self.data.eigen[x0[0], x1[1], x0[2]],
                       self.data.eigen[x1[0], x1[1], x0[2]],
                       self.data.eigen[x0[0], x0[1], x1[2]],
                       self.data.eigen[x1[0], x0[1], x1[2]],
                       self.data.eigen[x0[0], x1[1], x1[2]],
                       self.data.eigen[x1[0], x1[1], x1[2]])
        
        normal = trilerp(x, x0, x1, 
                       self.data.normal[x0[0], x0[1], x0[2]],
                       self.data.normal[x1[0], x0[1], x0[2]],
                       self.data.normal[x0[0], x1[1], x0[2]],
                       self.data.normal[x1[0], x1[1], x0[2]],
                       self.data.normal[x0[0], x0[1], x1[2]],
                       self.data.normal[x1[0], x0[1], x1[2]],
                       self.data.normal[x0[0], x1[1], x1[2]],
                       self.data.normal[x1[0], x1[1], x1[2]])
        
        return SGGXVoxel(albedo, density, eigen, normal)

    @ti.func
    def Tr(self):
        pass

    @ti.func
    def specular_phase_func(self):
        pass

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
    def ndf(self):
        """
        Normal distribution function
        """
        pass

    @ti.func
    def sigma(self, w: vec3):
        pass

    @ti.func
    def sggx_matrix(self, x: vec3, ) -> mat3:
        voxel = self.at(x)
        
        pass
