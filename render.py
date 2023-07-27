import taichi as ti
from tqdm import tqdm
from taichi.math import vec3, exp, distance

from camera import PerspectiveCamera
from envmap import Envmap
from volume import Volume
from utils import sample_uniform_sphere, Ray, AABB
from sggx import SGGX



@ti.func
def Tr(x: vec3, y: vec3, density):
    """
    Calculate Transmittance between x and y, assuming constant density.
    """
    return exp(-distance(x,y)*density)

@ti.func
def intersect_edge(x: vec3, aabb: AABB) -> bool:
    """
    Determine whether the ray intersects with AABB's edges.
    """
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
def intersect_aabb(ray: Ray, aabb: AABB) -> tuple[float, float, bool]:
    """
    Ray AABB intersection. 
    https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    """
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
            tmin, _, hit = intersect_aabb(ray, aabb)
            its = ray.o + tmin * ray.d
            if hit:
                hit_edge = intersect_edge(its, aabb)
                if hit_edge:
                    self.output[i, j] += vec3(1) / self.spp
                else:
                    self.output[i, j] += vec3(0.1, 0.8, 0.65) / self.spp
            else:
                self.output[i, j] += self.envmap.eval(ray) / self.spp


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

        self.output = ti.Vector.field(3, dtype=float, shape=self.cam.size) # Default init to 0
    
    def render(self):
        for _ in tqdm(range(self.spp), desc="Rendering"):
            self.render_sample()

    @ti.kernel
    def render_sample(self):
        """
        Render 1 sample per pixel. 
        """
        aabb = self.vol.get_aabb()
        for i, j in self.output:
            ray = self.cam.gen_ray(i, j)
            tmin, tmax, hit = intersect_aabb(ray, aabb)
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
                        tmin_s, tmax_s, _ = intersect_aabb(ray, aabb)
                        left_s = ray_s.o + tmin_s * ray_s.d
                        right_s = left_s
                        step_s = 1.0 / (self.N_samples_3 - 1) * (tmax_s - tmin_s)
                        for s3 in range(self.N_samples_3 - 1): # Inner transmittance
                            right_s = left_s + step_s * ray_s.d
                            voxel_s = self.vol.at(right_s)
                            transmittance_s *= Tr(left_s, right_s, voxel_s.density)
                            left_s = right_s
                        L_s += transmittance_s * self.envmap.eval(ray_s) / self.N_samples_2

                    color += transmittance * alpha * voxel.albedo * L_s
                    transmittance *= Tr(left, right, voxel.density)

                    left = right

                self.output[i, j] += (color + transmittance * self.envmap.eval(ray)) / self.spp
            else:
                self.output[i, j] += self.envmap.eval(ray) / self.spp


@ti.data_oriented
class SGGXIntegrator:
    def __init__(self, 
                 cam: PerspectiveCamera, 
                 envmap: Envmap, 
                 sggx: SGGX, 
                 spp: int, 
                 N_samples_1: int, 
                 N_samples_2: int,
                 N_samples_3: int) -> None:
        self.cam = cam
        self.envmap = envmap
        self.sggx = sggx
        self.spp = spp
        self.N_samples_1 = N_samples_1
        self.N_samples_2 = N_samples_2
        self.N_samples_3 = N_samples_3

        self.output = ti.Vector.field(3, dtype=float, shape=self.cam.size) # Default init to 0
    
    def render(self):
        for _ in tqdm(range(self.spp), desc="Rendering"):
            self.render_sample()

    @ti.kernel
    def render_sample(self):
        """
        Render 1 sample per pixel. 
        """
        aabb = self.sggx.get_aabb()
        for i, j in self.output:
            ray = self.cam.gen_ray(i, j)
            tmin, tmax, hit = intersect_aabb(ray, aabb)
            if (hit):
                color = vec3(0.0, 0.0, 0.0)
                transmittance = vec3(1.0, 1.0, 1.0)
                left = ray.o + tmin * ray.d
                right = left
                step = 1.0 / (self.N_samples_1 - 1) * (tmax - tmin)
                for s1 in range(self.N_samples_1 - 1): # Outer integral
                    right = left + step * ray.d
                    voxel = self.sggx.at(right)
                    S_xx, S_yy, S_zz, S_xy, S_xz, S_yz = voxel.S_mat[0,0], voxel.S_mat[1,1], voxel.S_mat[2,2], voxel.S_mat[0,1], voxel.S_mat[0,2], voxel.S_mat[1,2]
                    sigma_t = self.sggx.sigma(-ray.d, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz) * voxel.density
                    alpha = 1 - exp(-step*sigma_t)
                    L_s = vec3(0.0, 0.0, 0.0) 
                    for s2 in range(self.N_samples_2): # Inner integral
                        transmittance_s = vec3(1.0, 1.0, 1.0)
                        ray_s = Ray(o=right, d=self.sggx.sample_specular(-ray.d, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz))
                        tmin_s, tmax_s, _ = intersect_aabb(ray, aabb)
                        left_s = ray_s.o + tmin_s * ray_s.d
                        right_s = left_s
                        step_s = 1.0 / (self.N_samples_3 - 1) * (tmax_s - tmin_s)
                        for s3 in range(self.N_samples_3 - 1): # Inner transmittance
                            right_s = left_s + step_s * ray_s.d
                            voxel_s = self.sggx.at(right_s)
                            S_xx_s, S_yy_s, S_zz_s, S_xy_s, S_xz_s, S_yz_s = voxel_s.S_mat[0,0], voxel_s.S_mat[1,1], voxel_s.S_mat[2,2], voxel_s.S_mat[0,1], voxel_s.S_mat[0,2], voxel_s.S_mat[1,2]
                            sigma_t_s = self.sggx.sigma(-ray_s.d, S_xx_s, S_yy_s, S_zz_s, S_xy_s, S_xz_s, S_yz_s) * voxel_s.density
                            transmittance_s *= Tr(left_s, right_s, sigma_t_s)
                            left_s = right_s
                        L_s += transmittance_s * self.envmap.eval(ray_s) / self.N_samples_2

                    color += transmittance * alpha * voxel.albedo * L_s
                    transmittance *= Tr(left, right, sigma_t)

                    left = right

                self.output[i, j] += (color + transmittance * self.envmap.eval(ray)) / self.spp
            else:
                self.output[i, j] += self.envmap.eval(ray) / self.spp

