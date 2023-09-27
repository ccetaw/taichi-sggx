import taichi as ti
from taichi.math import vec3, clamp, exp

from utils import trilerp, inverse_cdf, sample_uniform_sphere, AABB, Voxel, Ray,  vecD, sd_bunny, laplacian_cdf


@ti.data_oriented
class Volume:

    def __init__(self, lb: vec3, rt: vec3, sigma_s: vec3, sigma_t: float, res: int, hetero: bool = False, lerp: bool =False) -> None:
        self.aabb = AABB(lb=lb, rt=rt) # In local space, aabb assumed to be [0,0,0] and [1,1,1]
        self.res = res
        self.units = (rt - lb) / (res - 1)
        self.lerp=lerp
        self.sigma_s = sigma_s
        self.sigma_t = sigma_t
        self.albedo = self.sigma_s / self.sigma_t
        self.data = Voxel.field(shape=(res, res, res))
        if hetero:
            self.gen_gradient_volume()
        else:
            self.data.density.fill(1)
    
    @ti.func
    def index2xyz(self, i, j, k):
        return self.aabb.lb + vec3(i, j, k) * self.units

    @ti.func
    def density(self, x: vec3) -> float:
        """
        Density at x. Using nearest neighbor or trilinear interpolation
        """
        x = clamp(x, xmin=self.aabb.lb, xmax=self.aabb.rt-0.001)
        x -= self.aabb.lb
        x /= self.aabb.rt - self.aabb.lb
        x = x * (self.res - 1)
        x0 = ti.cast(x, int)
        density = 1.0
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
            
        else:
            density = self.data.density[x0]

        return density


    @ti.func
    def sample(self, tmin: float, tmax: float, ray: Ray, n_samples: int):
        # n_samples <= 32
        rayPDF = vecD(0.0)
        interval = (tmax - tmin) / (n_samples - 1)
        near = ray.o + tmin * ray.d

        scatter = False
        t = 0.0
        beta = vec3(1.0)
        Tr = 1.0
        for i in range(n_samples - 1):
            sigma = self.density(near + interval*(i + 0.5)*ray.d) * self.sigma_t # Using the value at the middle point 
            alpha = 1 - exp(-sigma * interval)
            rayPDF[i] = alpha * Tr
            Tr *= 1 - alpha
        
        rayPDF[n_samples - 1] = Tr # Probability of no volume scattering
        rayPDF = rayPDF/rayPDF.sum()

        t = tmin + inverse_cdf(rayPDF, n_samples) * (tmax - tmin) * n_samples / (n_samples - 1)
        if t < tmax: # Volume scattering
            scatter = True
            beta = self.albedo
        
        return scatter, t, beta


    @ti.func
    def Tr(self, tmin: float, tmax: float, ray: Ray, n_samples: int) -> vec3:
        interval = (tmax - tmin) / (n_samples - 1)
        near = ray.o + tmin * ray.d

        Tr = vec3(1.0)

        for i in range(n_samples - 1):
            sigma = self.density(near + interval*(i + 0.5)*ray.d) * self.sigma_t # Using the value at the middle point 
            Tr *= exp(-sigma * interval)

        return Tr

    @ti.func
    def sample_p(self, x: vec3, wi: vec3):
        wo = sample_uniform_sphere()
        return wo
    
    @ti.func
    def get_sigma_t(self, x: vec3, wi: vec3):
        return self.density(x) * self.sigma_t
    
    @ti.func
    def get_albedo(self, x: vec3):
        return self.albedo
        
    @ti.func
    def get_aabb(self):
        """
        Return AABB of the volume in world space
        """
        lb = self.aabb.lb
        rt = self.aabb.rt
        return AABB(lb=lb, rt=rt)

    @ti.kernel
    def gen_gradient_volume(self):
        for i,j,k in self.data:
           self.data.density[i, j, k]  = 5 * laplacian_cdf(-sd_bunny(self.index2xyz(i, j, k)))
           # self.data.density[i, j, k] =  k/self.res
