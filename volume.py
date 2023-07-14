import taichi as ti
import numpy as np
from taichi.math import mat4, vec3, vec4, clamp, distance, exp

from utils import trilerp

@ti.dataclass
class Voxel:
    density: vec3
    albedo: vec3

@ti.dataclass
class AABB:
    lb: vec3
    rt: vec3


@ti.data_oriented
class Volume:

    def __init__(self, lb: vec3, rt: vec3, res: int, hetero: bool = False, lerp: bool =False) -> None:
        self.aabb = AABB(lb=lb, rt=rt) # In local space, aabb assumed to be [0,0,0] and [1,1,1]
        self.res = res
        self.lerp=lerp
        self.data = Voxel.field(shape=(res, res, res))
        if hetero:
            self.gen_gradient_volume()
        else:
            self.data.density.fill(0.7)
            self.data.albedo.fill(0.5)

    @ti.func 
    def at(self, x):
        """
        voxel data at world space position x
        """
        x = clamp(x, xmin=self.aabb.lb, xmax=self.aabb.rt-0.001)
        x -= self.aabb.lb
        x /= self.aabb.rt - self.aabb.lb
        x = x * (self.res - 1)
        x0 = ti.cast(x, int)
        voxel = Voxel()
        if self.lerp:
            x1 = x0 + 1
            density = trilerp(x, x0, x1, 
                        self.data.density[x0[0], x0[1], x0[2]],
                        self.data.density[x1[0], x0[1], x0[2]],
                        self.data.density[x0[0], x1[1], x0[2]],
                        self.data.density[x1[0], x1[1], x0[2]],
                        self.data.density[x0[0], x0[1], x1[2]],
                        self.data.density[x1[0], x0[1], x1[2]],
                        self.data.density[x0[0], x1[1], x1[2]],
                        self.data.density[x1[0], x1[1], x1[2]])
            
            albedo = trilerp(x, x0, x1, 
                        self.data.albedo[x0[0], x0[1], x0[2]],
                        self.data.albedo[x1[0], x0[1], x0[2]],
                        self.data.albedo[x0[0], x1[1], x0[2]],
                        self.data.albedo[x1[0], x1[1], x0[2]],
                        self.data.albedo[x0[0], x0[1], x1[2]],
                        self.data.albedo[x1[0], x0[1], x1[2]],
                        self.data.albedo[x0[0], x1[1], x1[2]],
                        self.data.albedo[x1[0], x1[1], x1[2]])
            voxel = Voxel(density=density, albedo=albedo)
        else:
            voxel = self.data[x0]

        return voxel

    @ti.func
    def get_aabb(self):
        """
        Return AABB of the volume in world space
        """
        lb = self.aabb.lb
        rt = self.aabb.rt
        return AABB(lb=lb, rt=rt)

    @ti.func 
    def Tr(self, x, y):
        """
        Assuming constant density value between x and y
        Take the value at x
        """
        return exp(-distance(x,y)*self.at(y).density)

    @ti.kernel
    def gen_gradient_volume(self):
        for i,j,k in self.data:
            self.data.albedo[i, j, k] = vec3(0.2, 0.1, 0.2) + k/self.res * vec3(0.8, 0.1, 0.8)
            self.data.density[i, j, k] =  k/self.res * vec3(1)
