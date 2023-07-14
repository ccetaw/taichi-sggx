import taichi as ti
import numpy as np
from taichi.math import vec3

from camera import PerspectiveCamera
from envmap import Envmap
from render import Integrator
from volume import Volume

ti.init(arch=ti.gpu, kernel_profiler=True)

c2w = np.array([[-np.sqrt(2)/2, 0, np.sqrt(2)/2, 2],
                [np.sqrt(2)/2, 0, np.sqrt(2)/2, 2],
                [0, 1, 0, 0],
                [0, 0, 0, 1]], dtype=float)

l2w = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]], dtype=float)

# v2w = np.array([[1, 0, 0, 0],
#                 [0, 1, 0, 0],
#                 [0, 0, 1, 0],
#                 [0, 0, 0, 1]], dtype=float)

v2w = np.array([[2, 0, 0, -1],
                [0, 2, 0, -1],
                [0, 0, 2, -1],
                [0, 0, 0, 1]], dtype=float)


camera = PerspectiveCamera(35, 800, 800, c2w)
envmap = Envmap('assets/Tokyo_BigSight_3k.hdr', l2w)
volume = Volume(v2w, 256, hetero=True)
integrator = Integrator(camera, envmap, volume, 32, 64, 32)

integrator.render()
ti.profiler.print_kernel_profiler_info('trace')
# integrator.render_image()

ti.tools.imshow(integrator.output)

save_as = "./output/heterogeneous.png"
ti.tools.imwrite(integrator.output, save_as)

