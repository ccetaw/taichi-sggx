import taichi as ti
import numpy as np
from taichi.math import vec3, rot_by_axis, translate

from camera import PerspectiveCamera
from envmap import Envmap
from render import VolIntegrator, AABBIntegrator, SGGXIntegrator
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

l2w = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]], dtype=float)

lb = vec3(-1,-1,-1)
rt = vec3(1,1,1)


camera = PerspectiveCamera(30, 800, 800, c2w)
envmap = Envmap('assets/limpopo_golf_course_3k.hdr', l2w)
# volume = Volume(lb, rt, 256, hetero=True, lerp=False)
sggx = SGGX(lb, rt, 128, lerp=False)
# integrator = AABBIntegrator(camera, envmap, volume, 32)
# integrator = VolIntegrator(camera, envmap, volume, 32, 64, 32, 16)
integrator = SGGXIntegrator(camera, envmap, sggx, 16, 64, 32, 16)


# Render
integrator.render()
ti.profiler.print_kernel_profiler_info('trace')

ti.tools.imshow(integrator.output)

# Save to png
save_as = "./test.png"
ti.tools.imwrite(integrator.output, save_as)

