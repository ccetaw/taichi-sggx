import taichi as ti
from tqdm import tqdm
from taichi.math import vec3, exp, min

from camera import PerspectiveCamera
from envmap import Envmap
from volume import Volume
from utils import Ray, AABB


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
                 vol: Volume) -> None:
        self.cam = cam
        self.envmap = envmap
        self.vol = vol

        self.output = ti.Vector.field(3, dtype=float, shape=self.cam.size)
    
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
                    self.output[i, j] += vec3(1)
                else:
                    self.output[i, j] += vec3(0.1, 0.8, 0.65)
            else:
                self.output[i, j] += self.envmap.eval(ray)


@ti.data_oriented
class SingleIntegrator:
    def __init__(self, 
                 cam: PerspectiveCamera, 
                 envmap: Envmap, 
                 vol, 
                 N_samples_1: int, 
                 N_samples_2: int,
                 N_samples_3: int) -> None:
        self.cam = cam
        self.envmap = envmap
        self.vol = vol
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
        Render 1 sample per pixel. Single scattering model.
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
                    alpha = 1 - exp(-step*self.vol.get_sigma_t((right+left)/2, -ray.d))
                    L_s = vec3(0.0, 0.0, 0.0) 
                    for s2 in range(self.N_samples_2): # Inner integral
                        transmittance_s = vec3(1.0, 1.0, 1.0)
                        ray_s = Ray(o=(right+left)/2, d=self.vol.sample_p((right+left)/2, -ray.d))
                        tmin_s, tmax_s, _ = intersect_aabb(ray, aabb)
                        transmittance_s = self.vol.Tr(tmin_s, tmax_s, ray_s, self.N_samples_3)
                        L_s += transmittance_s * self.envmap.eval(ray_s) / self.N_samples_2

                    color += transmittance * alpha * self.vol.get_albedo((right+left)/2) * L_s
                    transmittance *= 1 - alpha

                    left = right

                self.output[i, j] += (color + transmittance * self.envmap.eval(ray))
            else:
                self.output[i, j] += self.envmap.eval(ray)


@ti.data_oriented
class PathIntegrator:
    def __init__(self, 
                 cam: PerspectiveCamera, 
                 envmap: Envmap, 
                 vol: Volume, 
                 max_depth: int, 
                 ruassian_roulette_start: int,
                 n_samples: int) -> None:
        self.cam = cam
        self.envmap = envmap
        self.vol = vol
        self.max_depth = max_depth
        self.ruassian_roullete_start = ruassian_roulette_start
        self.n_samples = n_samples

        self.output = ti.Vector.field(3, dtype=float, shape=self.cam.size) # Default init to 0

    def render(self):
        for _ in tqdm(range(self.spp), desc="Rendering"):
            self.render_sample()

    @ti.kernel
    def render_sample(self):
        """
        Path tracing. Use ray marching to sample medium interaction point. 
        Only material sampling.
        """
        aabb = self.vol.get_aabb()
        for i, j in self.output:
            ray = self.cam.gen_ray(i, j)
            tmin, tmax, hit = intersect_aabb(ray, aabb)
            if hit:
                # Run path tracing
                L = vec3(0.0)
                beta = vec3(1.0)
                bounces = 0
                while(True and bounces < self.max_depth):
                    L += beta * self.vol.Tr(tmin, tmax, ray, self.n_samples) * self.envmap.eval(ray)

                    # Ruassian roulette termination
                    rp = 1.0
                    if bounces > self.ruassian_roullete_start:
                        rp = min(beta.max(), 0.99)
                    if (rp == 0.0 or ti.random() > rp):
                        break
                    else:
                        beta /= rp
                    
                    # Sample next ray
                    scatter, t, weight = self.vol.sample(tmin, tmax, ray, self.n_samples)
                    if scatter:
                        p = ray.o + t * ray.d
                        wo = self.vol.sample_p(p, -ray.d)
                        beta *= weight
                        ray = Ray(p, wo)
                        tmin, tmax, hit = intersect_aabb(ray, aabb)
                        if tmax < 0.001:
                            break
                    else:
                        break
                    bounces += 1

                self.output[i, j] += L

            else:
                self.output[i, j] += self.envmap.eval(ray)
