.. image:: https://anaconda.org/conda-forge/levelsetpy/badges/version.svg
   :target: https://anaconda.org/conda-forge/levelsetpy

.. image:: https://img.shields.io/pypi/v/control.svg
   :target: https://pypi.org/project/levelsetpy/

.. image:: https://github.com/python-control/python-control/actions/workflows/python-package-conda.yml/badge.svg
   :target: https://github.com/python-control/python-control/actions/workflows/python-package-conda.yml

.. image:: https://github.com/python-control/python-control/actions/workflows/install_examples.yml/badge.svg
   :target: https://github.com/python-control/python-control/actions/workflows/install_examples.yml

.. image:: https://github.com/python-control/python-control/actions/workflows/control-slycot-src.yml/badge.svg
   :target: https://github.com/python-control/python-control/actions/workflows/control-slycot-src.yml

.. image:: https://coveralls.io/repos/python-control/python-control/badge.svg
   :target: https://coveralls.io/r/python-control/python-control
   
### LevelSetsPy Library

This repo implements basic level set methods. It started as a reimplementation if Ian Mitchell's level set toolbox,but has since metamorphorsed into a standalone project of its own. It was a project started by Lekan Molu in August, 2021.

#### Features

- Linear input/output systems in state-space and frequency domain
- Block diagram algebra: serial, parallel, and feedback interconnections
- Time response: initial, step, impulse
- Frequency response: Bode and Nyquist plots
- Control analysis: stability, reachability, observability, stability margins
- Control design: eigenvalue placement, linear quadratic regulator
- Estimator design: linear quadratic estimator (Kalman filter)



### Status

 [x] Testing.


### TODO's

- [+] Fix len of smooth in derivL computation in upwindFirstWENO5.py

- [ ] Tensor operators in PODS Function.
  - Todo Fall 2022.

  #### Derivatives Integration LF with Bugs
  - [+] upwindFirstWENO5a

#### Derivatives Integration via Lax Friedrichs
- [+] upwindFirstENO2
- [+] upwindFirstWENO5a
- [+] termLaxFriedrichs
- [+] termRestrictUpdate
- [+] artificialDissipationGLF


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