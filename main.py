import taichi as ti
import numpy as np
from taichi.math import vec3

from camera import PerspectiveCamera
from envmap import Envmap
from render import SingleIntegrator, PathIntegrator, AABBIntegrator
from volume import Volume
from sggx import SGGX

ti.init(arch=ti.gpu, kernel_profiler=True)

# Init scene parameters
c2w = np.array([[-np.sqrt(2)/2, -np.sqrt(6)/6, np.sqrt(3)/3, 2],
                [np.sqrt(2)/2, -np.sqrt(6)/6, np.sqrt(3)/3, 2],
                [0, np.sqrt(6)/3, np.sqrt(3)/3, 2],
                [0, 0, 0, 1]], dtype=float)

# c2w = np.array([[-np.sqrt(2)/2, 0, np.sqrt(2)/2, 2],
#                 [np.sqrt(2)/2, 0, np.sqrt(2)/2, 2],
#                 [0, 1, 0, 0],
#                 [0, 0, 0, 1]], dtype=float)

# c2w = np.array([[-1, 0, 0, 0],
#                 [0, 1, 0, 0],
#                 [0, 0, -1, -2],
#                 [0, 0, 0, 1]], dtype=float)

# c2w = np.array([[-1, 0, 0, 0],
#                 [0, -1, 0, 0],
#                 [0, 0, 1, 2],
#                 [0, 0, 0, 1]], dtype=float)

l2w = np.array([[-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]], dtype=float)

lb = vec3(-1,-1,-1)
rt = vec3(1,1,1)

sigma_s = vec3(0.8, 0.5, 0.5)
sigma_t = 1

camera = PerspectiveCamera(fov=30, width=800, height=800, c2w=c2w)

envmap = Envmap(path='assets/limpopo_golf_course_3k.hdr', l2w=l2w)

volume = Volume(lb=lb, rt=rt, 
                sigma_s=sigma_s, 
                sigma_t=sigma_t, 
                res=256, 
                hetero=False, 
                lerp=True)

sggx = SGGX(lb=lb, rt=rt, 
            ks=0,
            res=128, 
            lerp=False)

# integrator = AABBIntegrator(camera, envmap, volume, 32)

# integrator = VolSingleIntegrator(camera, envmap, sggx, 32, 64, 32, 16)

# integrator = VolPathIntegrator(cam=camera,
#                                envmap=envmap,
#                                vol=volume,
#                                spp=256,
#                                max_depth=100,
#                                ruassian_roulette_start=10,
#                                n_samples=32)

integrator = PathIntegrator(cam=camera,
                               envmap=envmap,
                               vol=sggx,
                               spp=256,
                               max_depth=100,
                               ruassian_roulette_start=10,
                               n_samples=32)


# Render
integrator.render()
ti.profiler.print_kernel_profiler_info('trace')

ti.tools.imshow(integrator.output)

# Save to png
save_as = "./test2.png"
ti.tools.imwrite(integrator.output, save_as)

