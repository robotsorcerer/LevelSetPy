### LevelSets Py

This repo is a reproduction of Ian Mitchell's LevelSets Toolbox in Python written by Lekan Molu in August, 2021

### Demos

The [Tests](/Tests) folder contains examples of running basic tests. For example, you can view a demo of the grids creation by running

`python Tests/test_grids.py`


### TODO's

- [+] Fix len of smooth in derivL computation in upwindFirstWENO5.py

- [ ] data_proj is a lotta issue.
  - This is particular to Dr. Herbert's HelperOC class. Could not faithfully reproduce interpolation scheme in python.

- [ ] Tensor operators in PODS Function.
  - Todo Fall 2022.

- [ ] Write POD projections and reprojections for lifting dynamics.
  - Todo with Boris Kramer and Dr. Herbert.


  #### Derivatives Integration LF with Bugs
  - [+] upwindFirstWENO5a

#### Derivatives Integration LF
- [+] upwindFirstENO2
- [+] upwindFirstWENO5a
- [+] termLaxFriedrichs
- [+] termRestrictUpdate
- [+] artificialDissipationGLF
