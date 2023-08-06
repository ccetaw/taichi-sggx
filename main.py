import taichi as ti
import numpy as np
from taichi.math import vec3

from camera import PerspectiveCamera
from envmap import Envmap
from render import SingleIntegrator, PathIntegrator, AABBIntegrator
from volume import Volume
from sggx import SGGX
from utils import to_image

ti.init(arch=ti.gpu, kernel_profiler=True, device_memory_fraction=0.8)

# Camera setting
c2w = np.array([[-np.sqrt(2)/2, -np.sqrt(6)/6, np.sqrt(3)/3, 2],
                [np.sqrt(2)/2, -np.sqrt(6)/6, np.sqrt(3)/3, 2],
                [0, np.sqrt(6)/3, np.sqrt(3)/3, 2],
                [0, 0, 0, 1]], dtype=float)
fov = 30
width = 800 
height = 800
camera = PerspectiveCamera(fov=30, width=800, height=800, c2w=c2w)

# Envmap setting
path = 'assets/limpopo_golf_course_3k.hdr'
l2w = np.array([[0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]], dtype=float)
envmap = Envmap(path=path, l2w=l2w)

# Volume setting
lb = vec3(-1,-1,-1)
rt = vec3(1,1,1)
sigma_s = vec3(0.8, 0.5, 0.5)
sigma_t = 1
res = 256
hetero = False
lerp = True
ks = 0.5
# volume = Volume(lb=lb, rt=rt, 
#                 sigma_s=sigma_s, 
#                 sigma_t=sigma_t, 
#                 res=res, 
#                 hetero=hetero, 
#                 lerp=lerp)
volume = SGGX(lb=lb, rt=rt, 
            ks=ks,
            res=res, 
            lerp=lerp)

# Integrator setting
spp = 256
max_depth = 100
ruassian_roulette_start = 10
n_samples = 32

# integrator = AABBIntegrator(cam=camera, 
#                             envmap=envmap, 
#                             vol=volume)
# integrator = SingleIntegrator(cam=camera,
#                               envmap=envmap,
#                               vol=volume,
#                               N_samples_1=32,
#                               N_samples_2=32,
#                               N_samples_3=32)

integrator = PathIntegrator(cam=camera,
                            envmap=envmap,
                            vol=volume,
                            max_depth=max_depth,
                            ruassian_roulette_start=ruassian_roulette_start,
                            n_samples=n_samples)

window = ti.ui.Window("Renderer", (width, height))
canvas = window.get_canvas()


i = 1
save_as = "./output/path_sggx.png"
img = ti.Vector.field(3, dtype=float, shape=[width, height])
while window.running:
    if i <= spp:
        integrator.render_sample()
        to_image(img, integrator.output, i)
        canvas.set_image(img)
    elif i == spp + 1:
        print(f"Image rendered. Saved to {save_as}.")
        ti.tools.imwrite(img, save_as)
        canvas.set_image(img)
    else:
        canvas.set_image(img)
    i += 1
    window.show()

