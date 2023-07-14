import numpy as np
import taichi as ti
from taichi.math import vec2, vec3, mat3, acos, atan2, pi, clamp

from utils import bilerp
from camera import Ray


@ti.data_oriented
class Envmap:
    def __init__(self, path: str, l2w: np.ndarray):
        img = ti.tools.imread(path).astype(np.float32)
        self.img = vec3.field(shape=img.shape[:2])
        self.img.from_numpy(img / 255)
        self.l2w = mat3(l2w) # light to world transform matrix, rotation only
        self.w2l = mat3(np.linalg.inv(l2w)) 
        
    @ti.func
    def eval(self, ray: Ray) -> vec3:
        d = ray.d

        w = self.w2l @ d

        theta = acos(w.z)
        phi = atan2(w.y, w.x)
        if (phi<0):
            phi += 2 * pi

        x = phi * 0.5 / pi # in [0, 1]
        y = 1- theta / pi  # in [0, 1], invert y axis
        x = clamp(x, 0, 0.999) * (self.img.shape[0]-1) # avoid out of range
        y = clamp(y, 0, 0.999) * (self.img.shape[1]-1)

        x0 = int(x)
        y0 = int(y)
        x1 = x0 + 1
        y1 = y0 + 1
        
        return bilerp(vec2(x,y), vec2(x0,y0), vec2(x1,y1), self.img[x0,y0], self.img[x1,y0], self.img[x0,y1], self.img[x1, y1])
    
    # Taichi loads image with following convetion
    # -------> x:width
    # |
    # |
    # y: height

