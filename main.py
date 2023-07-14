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

lb = vec3(-1,-1,-1)
rt = vec3(1,1,1)


camera = PerspectiveCamera(35, 800, 800, c2w)
envmap = Envmap('assets/limpopo_golf_course_3k.hdr', l2w)
volume = Volume(lb, rt, 256, hetero=True, lerp=False)
integrator = Integrator(camera, envmap, volume, 32, 64, 32)

integrator.render()
ti.profiler.print_kernel_profiler_info('trace')

ti.tools.imshow(integrator.output)

save_as = "./output/heterogeneous.png"
ti.tools.imwrite(integrator.output, save_as)

