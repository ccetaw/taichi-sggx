import taichi as ti
import numpy as np
from taichi.math import vec3

from camera import PerspectiveCamera
from envmap import Envmap
from render import Integrator
from volume import Volume
from utils import image_camera2taichi

ti.init(arch=ti.gpu)

c2w = np.array([[-1, 0, 0, 0],
                [0, 0, 1, -2],
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


camera = PerspectiveCamera(50, 800, 800, c2w)
envmap = Envmap('assets/limpopo_golf_course_3k.hdr', l2w)
volume = Volume(v2w, 128)
integrator = Integrator(camera, envmap, volume, 128, 64)

integrator.render()

image_taichi = ti.Vector.field(3, dtype=float, shape=camera.size)
image_camera2taichi(integrator.output, image_taichi)

ti.tools.imshow(image_taichi)

