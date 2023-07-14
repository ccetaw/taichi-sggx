import taichi as ti
import numpy as np
from taichi.math import vec3, max, min, exp

from camera import PerspectiveCamera, Ray
from envmap import Envmap
from volume import Volume, AABB
from utils import sample_uniform_sphere

@ti.data_oriented
class Integrator:

    def __init__(self, 
                 cam: PerspectiveCamera, 
                 envmap: Envmap, 
                 vol: Volume, 
                 spp: int, 
                 N_samples_1: int, 
                 N_samples_2: int) -> None:
        self.cam = cam
        self.envmap = envmap
        self.vol = vol
        self.spp = spp
        self.N_samples_1 = N_samples_1
        self.N_samples_2 = N_samples_2

        self.points_1 = ti.Vector.field(3, dtype=float, shape=self.N_samples_1)
        self.points_2 = ti.Vector.field(3, dtype=float, shape=self.N_samples_2)
        self.output = ti.Vector.field(3, dtype=float, shape=self.cam.size)

    @ti.kernel
    def render(self):
        for i, j in self.output:
            for _ in range(self.spp):
                ray = self.cam.gen_ray(i, j)
                tmin, tmax, hit = self.intersect_aabb(ray, self.vol.get_aabb())
                if (hit):
                    color = vec3(0.0, 0.0, 0.0)
                    transmittance = vec3(1.0, 1.0, 1.0)
                    left = ray.o + tmin * ray.d
                    right = left
                    step = 1.0 / (self.N_samples_1 - 1) * (tmax - tmin)
                    for s1 in range(self.N_samples_1 - 1): # Outer integral
                        right = left + step * ray.d
                        voxel = self.vol.at(right)
                        alpha = 1 - exp(-step*voxel.density)
                        L_s = vec3(0.0, 0.0, 0.0) 
                        for s2 in range(self.N_samples_2): # Inner integral
                            transmittance_s = vec3(1.0, 1.0, 1.0)
                            ray_s = Ray(o=right, d=sample_uniform_sphere())
                            tmin_s, tmax_s, hit_s = self.intersect_aabb(ray, self.vol.get_aabb())
                            left_s = ray_s.o + tmin_s * ray_s.d
                            right_s = left_s
                            step_s = 1.0 / (self.N_samples_1 - 1) * (tmax_s - tmin_s)
                            for _s2 in range(self.N_samples_1 - 1):
                                right_s = left_s + step_s * ray_s.d
                                transmittance_s *= self.vol.Tr(left_s, right_s)
                                left_s = right_s
                            L_s = transmittance_s * self.envmap.eval(ray_s)

                        color += transmittance * alpha * voxel.albedo * L_s
                        transmittance *= self.vol.Tr(left, right)

                        left = right

                    self.output[i, j] += (color + transmittance * self.envmap.eval(ray)) / self.spp
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

        return max(0, tmin), tmax, hit # note the case that ray origin inside aabb

