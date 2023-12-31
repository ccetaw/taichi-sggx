import taichi as ti
from taichi.math import mat3, mat4, vec4, vec3, vec2, normalize
import numpy as np

from utils import Ray


@ti.data_oriented
class PerspectiveCamera:
    """
    Camera model following OpenGL convention, i.e. z axis face away from the scene.
    """

    def __init__(self, fov: float, width: int, height: int, c2w: np.ndarray) -> None:

        # Python scope variables
        self.size = [width, height]

        # Taichi scope variables
        # Scalars here are in python scope thus can not be furthur assigned values
        self.width = width
        self.height = height
        self.fov = fov
        self.output_size = vec2(width, height)
        self.inv_output_size = 1 / self.output_size
        self.c2w = mat4(c2w)
        self.focal = 0.5 * self.output_size.x / np.tan(np.deg2rad(self.fov))
        self.intrinsics = mat3(-self.focal, 0, self.output_size.x / 2,
                               0, -self.focal, self.output_size.y / 2,
                               0, 0, 1)
        
    @ti.func
    def gen_ray(self, i, j) -> Ray:
        o = self.c2w @ vec4(0,0,0,1)
        d = self.c2w @ vec4(i + 2*(ti.random() - 0.5) - self.output_size.x/2.0, j + 2*(ti.random() - 0.5) - self.output_size.y/2.0, -self.focal, 0)

        # print(i)
        
        origin = vec3(o[0], o[1], o[2])
        direction = normalize(vec3(d[0], d[1], d[2]))

        ray = Ray(o = origin, d = direction)
        # print(ray.d)

        return ray # Ray in world space
        
        

        
        
        