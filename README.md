# SGGX Microfacet Model 

This repo contains a simplest implementation of volume rendering including
- heterogeneous(homogeneous) volume rendering
- SGGX volume rendering, please refer to [The SGGX Microfacet Distribution](https://research.nvidia.com/publication/2015-08_sggx-microflake-distribution)

As it's for validation use only, only necessary components are included:
- perspective camera, with simple box reconstruction filter
- environment map
- voxel based volume representation
- transform could be applied on above components
- numerical integrator, no path tracing(might be added)

## Camera
We follow the OpenGL camera convention that the camera space is right-handed and the z-axis faces backwards. For convenience, we consider a image plane at focus and don't consider near and far clip. Please refer to [OpenGL Transformation](http://www.songho.ca/opengl/gl_projectionmatrix.html)

## Environment Map
We use an hdr image as environment map. The environment is considered placed so far away that rays with the same direction result in the same color. For more details, please refer to [PBR book](https://www.pbr-book.org/3ed-2018/Light_Sources/Infinite_Area_Lights). 

## Heterogeneous Volume

![](/output/heterogeneous.png)

We consider scattering, absorptive and non-emissive volume. For basics of volume rendering, please refer to [PBR book](https://www.pbr-book.org/3ed-2018/Volume_Scattering/Volume_Scattering_Processes)

For simplicity, the volume is confined in an AABB and the data is stored in voxels. Each voxel stores 2 3D vectors, standing for extinction coefficient and albedo. Trilinear interpolation is applied for points sampled in the volume. Also we consider uniform phase function, as we don't focus on heteroge

## SGGX Volume
[Work in Progess]

## Renderer
The overall rendering equation is 

$$\begin{aligned}L(\mathbf{x}, \omega) &= T_r(\mathbf{x}, \mathbf{x}_{\infty}(\omega, \mathbf{x}))L_e(\omega) \\ &+ \int_0^{z} T_r(\mathbf{x}, \mathbf{x}_t)\sigma_t(\mathbf{x}_t)\alpha(\mathbf{x}_t)\int_{S^2} f_p(\mathbf{x}_t,\omega,\omega')L_e(\omega') d\omega'dt
\end{aligned}$$

The radiance at point $\mathbf{x}$ from direction $\omega$ is composed of 2 terms. THe first term is the radiance from the background, with $T_r(\mathbf{x},\mathbf{y})$ being the transmittance between points $\mathbf{x}$ and $\mathbf{x}$, $L_e(\omega)$ being the radiance from the environment. As we mentioned, the environment radiance only depends on direction. And $\mathbf{x}_\infty(\omega)$ is the point at infinity in direction $\omega$ from point $\mathbf{x}$.

The second term is the accumulated scattering radiance. $sigma_t$ is the extincition coefficient, $\alpha$ is the albedo. In SGGX rendering, these two terms may depend on directions. The inner integration is the radiance from all directions scattered to direction $\omega$. $f_p$ is the phase function.
