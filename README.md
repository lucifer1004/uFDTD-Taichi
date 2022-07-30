# UFDTD-Taichi

> See also the [Julia](./julia/README.md) version.

*Understanding the Finite-Difference Time-Domain Method*, implemented in [Taichi](https://taichi-lang.org/).

## Chapter 3. Introduction to the FDTD method

### 1D Bare Bones

![1D Bare Bones](gif/1d_bare_bones.gif)

- PEC at the left boundary
- PMC at the right boundary
- Hardwired source at node 0

### 1D Additive

![1D Additive](gif/1d_additive.gif)

- PEC at the left boundary
- PMC at the right boundary
- Additive source at node 50

### 1D TFSF (Total Field / Scattered Field)

![1D TFSF](gif/1d_tfsf.gif)

- ABC at the left boundary
- ABC at the right boundary
- TFSF boundary between `hy[49]` and `ez[50]`

### 1D Dielectric

![1D Dielectric](gif/1d_dielectric.gif)

- ABC at the left boundary
- ABC at the right boundary (not working as expected due to non-unity relative permittivity)
- TFSF boundary between `hy[49]` and `ez[50]`
- A dielectric material starting at `ez[100]`

### 1D Lossy

![1D Lossy](gif/1d_lossy.gif)

- ABC at the left boundary
- TFSF boundary between `hy[49]` and `ez[50]`
- A lossy dielectric material starting at `ez[100]`
