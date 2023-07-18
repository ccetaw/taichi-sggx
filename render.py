import taichi as ti
from tqdm import tqdm
from taichi.math import vec3, max, min, exp

from camera import PerspectiveCamera, Ray
from envmap import Envmap
from volume import Volume, AABB
from utils import sample_uniform_sphere


@ti.data_oriented
class AABBIntegrator:
    def __init__(self, 
                 cam: PerspectiveCamera, 
                 envmap: Envmap,
                 vol: Volume,
                 spp: int) -> None:
        self.cam = cam
        self.envmap = envmap
        self.vol = vol
        self.spp = spp

        self.output = ti.Vector.field(3, dtype=float, shape=self.cam.size)
    
    def render(self):
        for _ in tqdm(range(self.spp), desc="Rendering"):
            self.render_sample()

    @ti.kernel
    def render_sample(self):
        aabb = self.vol.get_aabb()
        for i, j in self.output:
            ray = self.cam.gen_ray(i, j)
            tmin, _, hit = self.intersect_aabb(ray, aabb)
            its = ray.o + tmin * ray.d
            if hit:
                hit_edge = self.intersect_edge(its, aabb)
                if hit_edge:
                    self.output[i, j] += vec3(1) / self.spp
                else:
                    self.output[i, j] += vec3(0.1, 0.8, 0.65) / self.spp
            else:
                self.output[i, j] += self.envmap.eval(ray) / self.spp

    @ti.func
    def intersect_edge(self, x: vec3, aabb: AABB) -> bool:
        lb = aabb.lb
        rt = aabb.rt

        x0, x1, x2 = x[0], lb[0], rt[0]
        y0, y1, y2 = x[1], lb[1], rt[1]
        z0, z1, z2 = x[2], lb[2], rt[2]

        its_x, its_y, its_z = False, False, False
        its_edge = False

        if ti.abs(x0 - x1) < 1e-2 or ti.abs(x0 - x2) < 1e-2:
            its_x = True
        if ti.abs(y0 - y1) < 1e-2 or ti.abs(y0 - y2) < 1e-2:
            its_y = True
        if ti.abs(z0 - z1) < 1e-2 or ti.abs(z0 - z2) < 1e-2:
            its_z = True
        
        if its_x * its_y or its_x * its_z or its_y * its_z:
            its_edge = True

        return its_edge


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

@ti.data_oriented
class VolIntegrator:
    def __init__(self, 
                 cam: PerspectiveCamera, 
                 envmap: Envmap, 
                 vol: Volume, 
                 spp: int, 
                 N_samples_1: int, 
                 N_samples_2: int,
                 N_samples_3: int) -> None:
        self.cam = cam
        self.envmap = envmap
        self.vol = vol
        self.spp = spp
        self.N_samples_1 = N_samples_1
        self.N_samples_2 = N_samples_2
        self.N_samples_3 = N_samples_3

        self.output = ti.Vector.field(3, dtype=float, shape=self.cam.size)
    
    def render(self):
        for _ in tqdm(range(self.spp), desc="Rendering"):
            self.render_sample()

    @ti.kernel
    def render_sample(self):
        aabb = self.vol.get_aabb()
        for i, j in self.output:
            ray = self.cam.gen_ray(i, j)
            tmin, tmax, hit = self.intersect_aabb(ray, aabb)
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
                        tmin_s, tmax_s, _ = self.intersect_aabb(ray, aabb)
                        left_s = ray_s.o + tmin_s * ray_s.d
                        right_s = left_s
                        step_s = 1.0 / (self.N_samples_3 - 1) * (tmax_s - tmin_s)
                        for s3 in range(self.N_samples_3 - 1):
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



