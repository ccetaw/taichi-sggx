import taichi as ti
import numpy as np
from taichi.math import pi, vec3, mat3, clamp, sqrt, dot, cos, sin, normalize, pow

from volume import AABB
from utils import trilerp, build_orthonomal_basis, SGGXVoxel

@ti.data_oriented
class SGGX:

    def __init__(self, lb: vec3, rt: vec3, res: int, lerp: bool) -> None:
        self.aabb = AABB(lb=lb, rt=rt) # In local space, aabb assumed to be [0,0,0] and [1,1,1]
        self.res = res
        self.lerp = lerp
        self.data = SGGXVoxel.field(shape=(res, res, res))
        self.gen_uniform_sggx()

    @ti.kernel
    def gen_uniform_sggx(self):
        for i,j,k in self.data:
            self.data.albedo[i, j, k] = vec3(0.5, 0.5, 0.5)
            self.data.density[i, j, k] = 10
            self.data.S_mat[i, j, k] = mat3(0.1, 0, 0,
                                            0, 0.1, 0,
                                            0, 0, 1)

    @ti.func
    def at(self, x: vec3) -> SGGXVoxel:
        x = clamp(x, xmin=self.aabb.lb, xmax=self.aabb.rt-0.001)
        x -= self.aabb.lb
        x /= self.aabb.rt - self.aabb.lb
        x = x * (self.res - 1)
        x0 = ti.cast(x, int)
        voxel = SGGXVoxel()
        if self.lerp:
            x1 = x0 + 1
            albedo = trilerp(x, x0, x1, 
                        self.data.albedo[x0[0], x0[1], x0[2]],
                        self.data.albedo[x1[0], x0[1], x0[2]],
                        self.data.albedo[x0[0], x1[1], x0[2]],
                        self.data.albedo[x1[0], x1[1], x0[2]],
                        self.data.albedo[x0[0], x0[1], x1[2]],
                        self.data.albedo[x1[0], x0[1], x1[2]],
                        self.data.albedo[x0[0], x1[1], x1[2]],
                        self.data.albedo[x1[0], x1[1], x1[2]])
            
            density = trilerp(x, x0, x1, 
                        self.data.density[x0[0], x0[1], x0[2]],
                        self.data.density[x1[0], x0[1], x0[2]],
                        self.data.density[x0[0], x1[1], x0[2]],
                        self.data.density[x1[0], x1[1], x0[2]],
                        self.data.density[x0[0], x0[1], x1[2]],
                        self.data.density[x1[0], x0[1], x1[2]],
                        self.data.density[x0[0], x1[1], x1[2]],
                        self.data.density[x1[0], x1[1], x1[2]])
            
            S_mat = trilerp(x, x0, x1, 
                        self.data.S_mat[x0[0], x0[1], x0[2]],
                        self.data.S_mat[x1[0], x0[1], x0[2]],
                        self.data.S_mat[x0[0], x1[1], x0[2]],
                        self.data.S_mat[x1[0], x1[1], x0[2]],
                        self.data.S_mat[x0[0], x0[1], x1[2]],
                        self.data.S_mat[x1[0], x0[1], x1[2]],
                        self.data.S_mat[x0[0], x1[1], x1[2]],
                        self.data.S_mat[x1[0], x1[1], x1[2]])
            voxel = SGGXVoxel(albedo, density, S_mat)
        else:
            voxel = self.data[x0]
        
        return voxel

    @ti.func
    def get_aabb(self):
        """
        Return AABB of the volume in world space
        """
        lb = self.aabb.lb
        rt = self.aabb.rt
        return AABB(lb=lb, rt=rt)

    @ti.func
    def eval_specular(self, wi: vec3, wo: vec3, 
                      S_xx: float, S_yy: float, S_zz: float, 
                      S_xy: float, S_xz: float, S_yz: float):
        wh = normalize(wi + wo)
        return 0.25 * self.D(wh, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz) / self.sigma(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz)

    @ti.func
    def eval_diffuse(self, wi: vec3, wo: vec3,
                     S_xx: float, S_yy: float, S_zz: float, 
                     S_xy: float, S_xz: float, S_yz: float):
         wm = self.sample_VNDF(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz)
         return 1.0 / pi * max(0.0, dot(wo, wm))

    @ti.func
    def sample_specular(self, wi: vec3,
                        S_xx: float, S_yy: float, S_zz: float, 
                        S_xy: float, S_xz: float, S_yz: float) -> vec3:
        wm = self.sample_VNDF(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz)
        wo = -wi + 2.0 * wm * dot(wm, wi)
        return wo
    
    @ti.func
    def sample_diffuse(self, wi: vec3,
                       S_xx: float, S_yy: float, S_zz: float, 
                       S_xy: float, S_xz: float, S_yz: float) -> vec3:
        wm = self.sample_VNDF(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz)

        w1, w2 = build_orthonomal_basis(wm)

        # Uniform sample hemisphere
        u3 = ti.random()
        u4 = ti.random()

        r1 = 2.0 * u3 - 1.0
        r2 = 2.0 * u4 - 1.0

        phi, r = 0.0, 0.0
        if r1 == 0 and r2 == 0:
            r, phi = 0, 0
        elif r1*r1 > r2*r2:
            r = r1
            phi = (pi/4.0) * (r2/r1)
        else:
            r = r2
            phi = (pi/2) - (r1/r2) * (pi/4.0)
            
        x = r * cos(phi)
        y = r * sin(phi)
        z = sqrt(1 - x*x - y*y)
        wo = x*w1 + y*w2 + z*wm

        return wo

    @ti.func
    def sample_VNDF(self, wi: vec3,
                    S_xx: float, S_yy: float, S_zz: float, 
                    S_xy: float, S_xz: float, S_yz: float) -> vec3:
        u1 = ti.random()
        u2 = ti.random()

        r = sqrt(u1)
        phi = 2 * pi * u2
        u = r * cos(phi)
        v = r * sin(phi)
        w = sqrt(1 - u*u - v*v)

        wk, wj = build_orthonomal_basis(wi)

        # project S in this basis
        S_kk = wk.x*wk.x*S_xx + wk.y*wk.y*S_yy + wk.z*wk.z*S_zz + 2.0 * (wk.x*wk.y*S_xy + wk.x*wk.z*S_xz + wk.y*wk.z*S_yz)
        S_jj = wj.x*wj.x*S_xx + wj.y*wj.y*S_yy + wj.z*wj.z*S_zz + 2.0 * (wj.x*wj.y*S_xy + wj.x*wj.z*S_xz + wj.y*wj.z*S_yz)
        S_ii = wi.x*wi.x*S_xx + wi.y*wi.y*S_yy + wi.z*wi.z*S_zz + 2.0 * (wi.x*wi.y*S_xy + wi.x*wi.z*S_xz + wi.y*wi.z*S_yz)
        S_kj = wk.x*wj.x*S_xx + wk.y*wj.y*S_yy + wk.z*wj.z*S_zz + (wk.x*wj.y + wk.y*wj.x)*S_xy + (wk.x*wj.z + wk.z*wj.x)*S_xz + (wk.y*wj.z + wk.z*wj.y)*S_yz
        S_ki = wk.x*wi.x*S_xx + wk.y*wi.y*S_yy + wk.z*wi.z*S_zz + (wk.x*wi.y + wk.y*wi.x)*S_xy + (wk.x*wi.z + wk.z*wi.x)*S_xz + (wk.y*wi.z + wk.z*wi.y)*S_yz
        S_ji = wj.x*wi.x*S_xx + wj.y*wi.y*S_yy + wj.z*wi.z*S_zz + (wj.x*wi.y + wj.y*wi.x)*S_xy + (wj.x*wi.z + wj.z*wi.x)*S_xz + (wj.y*wi.z + wj.z*wi.y)*S_yz

        # compute normal
        sqrtDetSkji = sqrt(ti.abs(S_kk*S_jj*S_ii - S_kj*S_kj*S_ii - S_ki*S_ki*S_jj - S_ji*S_ji*S_kk + 2.0*S_kj*S_ki*S_ji))
        inv_sqrtS_ii = 1 / sqrt(S_ii)
        tmp = sqrt(S_jj * S_ii - S_ji * S_ji)
        Mk = vec3(sqrtDetSkji / tmp, 0.0, 0.0)
        Mj = vec3(-inv_sqrtS_ii*(S_ki*S_ji-S_kj*S_ii)/tmp , inv_sqrtS_ii*tmp, 0)
        Mi = vec3(inv_sqrtS_ii*S_ki, inv_sqrtS_ii*S_ji, inv_sqrtS_ii*S_ii)
        wm_kji =  normalize(u*Mk+v*Mj+w*Mi)

        # rotate back to world basis
        return  wm_kji.x * wk + wm_kji.y * wj + wm_kji.z * wi

    @ti.func
    def D(self, wm: vec3, 
          S_xx: float, S_yy: float, S_zz: float, 
          S_xy: float, S_xz: float, S_yz: float):
        """
        Normal distribution function
        """
        detS = S_xx*S_yy*S_zz - S_xx*S_yz*S_yz - S_yy*S_xz*S_xz - S_zz*S_xy*S_xy + 2.0*S_xy*S_xz*S_yz
        den = wm.x*wm.x*(S_yy*S_zz-S_yz*S_yz) + wm.y*wm.y*(S_xx*S_zz-S_xz*S_xz) + wm.z*wm.z*(S_xx*S_yy-S_xy*S_xy) + 2.0*(wm.x*wm.y*(S_xz*S_yz-S_zz*S_xy) + wm.x*wm.z*(S_xy*S_yz-S_yy*S_xz) + wm.y*wm.z*(S_xy*S_xz-S_xx*S_yz))
        result = pow(ti.abs(detS), 1.5) / (pi * den * den)
        return result

    @ti.func
    def sigma(self, wi: vec3, 
              S_xx: float, S_yy: float, S_zz: float, 
              S_xy: float, S_xz: float, S_yz: float) -> float:
        sigma_squared = wi.x*wi.x*S_xx + wi.y*wi.y*S_yy + wi.z*wi.z*S_zz + 2.0 * (wi.x*wi.y*S_xy + wi.x*wi.z*S_xz + wi.y*wi.z*S_yz)
        return sqrt(max(0, sigma_squared))
