import taichi as ti
import numpy as np
from taichi.math import mat4, vec3, vec4, clamp, distance, exp

from utils import trilerp

@ti.dataclass
class Voxel:
    sigma: vec3
    albedo: vec3

@ti.dataclass
class AABB:
    lb: vec3
    rt: vec3


@ti.data_oriented
class Volume:

    def __init__(self, v2w: np.ndarray, res: int) -> None:
        self.aabb = AABB(lt=vec3(0,0,0), rt=vec3(1,1,1)) # In local space, aabb assumed to be [0,0,0] and [1,1,1]
        self.v2w = mat4(v2w)
        self.w2v = mat4(np.linalg.inv(v2w))
        self.res = res
        
        self.data = Voxel.field(shape=(res, res, res))

    @ti.func 
    def at(self, x):
        tmp = vec4(x[0], x[1], x[2], 1)
        tmp = self.w2v @ tmp
        x = vec3(tmp.x, tmp.y, tmp.z)
        x = clamp(x, xmin=vec3([0,0,0]), xmax=vec3([0.999,0.999,0.999]))
        x = x * (self.res - 1)
        x0 = ti.cast(x, int)
        x1 = x0 + 1
        sigma = trilerp(x, x0, x1, 
                       self.data.sigma[x0[0], x0[1], x0[2]],
                       self.data.sigma[x1[0], x0[1], x0[2]],
                       self.data.sigma[x0[0], x1[1], x0[2]],
                       self.data.sigma[x1[0], x1[1], x0[2]],
                       self.data.sigma[x0[0], x0[1], x1[2]],
                       self.data.sigma[x1[0], x0[1], x1[2]],
                       self.data.sigma[x0[0], x1[1], x1[2]],
                       self.data.sigma[x1[0], x1[1], x1[2]])
        
        albedo = trilerp(x, x0, x1, 
                       self.data.albedo[x0[0], x0[1], x0[2]],
                       self.data.albedo[x1[0], x0[1], x0[2]],
                       self.data.albedo[x0[0], x1[1], x0[2]],
                       self.data.albedo[x1[0], x1[1], x0[2]],
                       self.data.albedo[x0[0], x0[1], x1[2]],
                       self.data.albedo[x1[0], x0[1], x1[2]],
                       self.data.albedo[x0[0], x1[1], x1[2]],
                       self.data.albedo[x1[0], x1[1], x1[2]])
        
        return Voxel(sigma=sigma, albedo=albedo)

    @ti.func
    def get_aabb(self):
        lb = self.aabb.lb
        rt = self.aabb.rt
        tmp1 = vec4(lb.x, lb.y, lb.z, 1)
        tmp2 = vec4(rt.x, rt.y, rt.z, 1)
        lb_w = self.v2w @ tmp1
        rt_w = self.v2w @ tmp2
        return AABB(lb=vec3(lb_w.x, lb_w.y, lb_w.z), rt=vec3(rt_w.x, rt_w.y, rt_w.z))

    @ti.func 
    def Tr(self, x, y):
        """
        Assuming constant density value between x and y
        Take the value at y
        """
        return exp(-distance(x,y)*self.at(y).sigma)

