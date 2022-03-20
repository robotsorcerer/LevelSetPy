### LevelSetsPy Library

This repo implements basic level set methods. It started as a reimplementation if Ian Mitchell's level set toolbox,but has since metamorphorsed into a standalone project of its own. It was a project started by Lekan Molu in August, 2021.

#### Features

- Explicit Hamiltonian Integration Schemes: Courant-Friedrichs Lax; Dissipasion; Lax-Friedrichs Schemes.
- Eikonal Equation: Signed-distance functions for level set representations of value functions.
- Optimization Libraries: ADMM, Chambolle-Pock TVD schemes.
- Upwinding: Directional approximations to  essentially non-oscillatory integration schemes.
- Value function visualizations in real-time: Marching cubes for 3D surface construction; Lewiner and Lorensen's algorithm.

### Status

 [x] Testing.


### TODO's

- [+] Fix len of smooth in derivL computation in upwindFirstWENO5.py

- [ ] Tensor operators in PODS Function.
  - Todo Fall 2022.

  #### Derivatives Integration LF with Bugs
  - [+] upwindFirstWENO5a


If you have used LevelSetPy in your work, please cite it:

```tex
@misc{LevelSetPy,
  author = {Ogunmolu, Olalekan},
  title = {{A Numerical Toolbox for the Scalable Analysis of Hamilton-Jacobi PDEs.}},
  year = {2022},
  howpublished = {\url{https://github.com/robotsorcerer/LevelSetPy}},
  note = {Accessed March 11, 2022}
}
```