import taichi as ti
import numpy as np
from taichi.math import vec3, max, min

from camera import PerspectiveCamera, Ray
from envmap import Envmap
from volume import Volume, AABB

@ti.data_oriented
class Integrator:

    def __init__(self, cam: PerspectiveCamera, envmap: Envmap, vol: Volume, spp: int, N_samples: int) -> None:
        self.cam = cam
        self.envmap = envmap
        self.vol = vol
        self.spp = spp
        self.N_samples = N_samples

        self.output = ti.Vector.field(3, dtype=float, shape=self.cam.size)

    @ti.kernel
    def render(self):
        for i, j in self.output:
            for _ in range(self.spp):
                ray = self.cam.gen_ray(i, j)
                tmin, tmax, hit = self.intersect_aabb(ray, self.vol.get_aabb())
                if (hit):
                    self.output[i, j] += vec3(0.8, 0.8, 0.8) / self.spp
                else:
                    self.output[i, j] += self.envmap.eval(ray) / self.spp

    @ti.func
    def intersect_aabb(self, ray: Ray, aabb: AABB) -> tuple[float, float, bool]:
        org = ray.o
        dir = ray.d
        inv_dir = 1 / (dir + 1e-6) # Avoid dividing by zero
        
        lb = aabb.lb
        rt = aabb.rt

        t1 = (lb.x - org.x) * inv_dir.x
        t2 = (rt.x - org.x) * inv_dir.x
        t3 = (lb.y - org.y) * inv_dir.y
        t4 = (rt.y - org.y) * inv_dir.y
        t5 = (lb.z - org.z) * inv_dir.z
        t6 = (rt.z - org.z) * inv_dir.z

        tmin = max(min(t1, t2), min(t3, t4), min(t5, t6))
        tmax = min(max(t1, t2), max(t3, t4), max(t5, t6))
        hit = False
        if tmax > 0 and tmax > tmin:
            hit = True

        return max(0, tmin), tmax, hit

    @ti.func
    def sample_points(self, ray: Ray, tmin: float, tmax: float):
        t_sample_np = np.linspace()
        pass