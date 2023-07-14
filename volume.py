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

    def __init__(self, v2w: np.ndarray, res: int, random: bool = False) -> None:
        self.aabb = AABB(lt=vec3(0,0,0), rt=vec3(1,1,1)) # In local space, aabb assumed to be [0,0,0] and [1,1,1]
        self.v2w = mat4(v2w)
        self.w2v = mat4(np.linalg.inv(v2w))
        self.res = res
        
        self.data = Voxel.field(shape=(res, res, res))
        if random:
            self.gen_gradient_volume()
        else:
            self.data.sigma.fill(1)
            self.data.albedo.fill(0.5)

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
        # print(sigma)
        
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
        tmp_x = vec4(x[0], x[1], x[2], 1)
        tmp_y = vec4(y[0], y[1], y[2], 1)
        tmp_x = self.w2v @ tmp_x
        tmp_y = self.w2v @ tmp_y
        x = vec3(tmp_x[0], tmp_x[1], tmp_x[2])
        y = vec3(tmp_y[0], tmp_y[1], tmp_y[2])
        return exp(-distance(x,y)*self.at(x).sigma)

    @ti.kernel
    def gen_gradient_volume(self):
        for i,j,k in self.data:
            self.data.albedo[i, j, k] = k/self.res * vec3(0.7, 0.7, 0.7)
            self.data.sigma[i, j, k] = vec3(0.7, 0.7, 0.1) + k/self.res * vec3(0.4, 0.1, 0.2)
