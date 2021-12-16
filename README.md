SCPMel : A Pytorch-friendly package providing tools for building 1D PINN models
================================================


## Contents
  - [Introduction](#introduction)
  - [Prerequisites](#prerequisites)
  - [Installation](#install)
  	- [Building the conda environnement](#building-the-conda-environnement)
  	- [Installing SCPMel](#installing-scpmel)
  - [References](#references)
---


## Introduction

SCPMel is a Pytorch-based package providing tools for building 1D Physically
Informed Neural Networks (PINN) models. SCPMel is composed of several submodules, 
using any of these submodule can be made through implicit or explicit import. For instance,

``` python
import scpmel.integrate
from scpmel import integrate
```
The following submodules are available:

| Submodule | Description |
| --------------  | --------------- |
| integrate     | A pytorch-based inplementation of explicit Runge-Kutta schemes following the **scipy.integrate._ivp** module.  |
| les               | Common filters and spectral routines used in Large Eddy Simulation (LES). |
| reduction    | A helper class that restrict any Pytorch optimizer to a subspace spawn by principal components. |
| sampling    | Custom sampling routines. |
| training       | Utility routines to ease training Pytorch models. |
| utils             | Additionals tools and plot utilities. |

Note that SCPMel is currently only supported  on Unix and has only been tested on Ubuntu 20.04.

---
## Prerequisites

SCPMel requires the following dependencies to be installed:

1. [Python](https://www.python.org/) >= 3.8.5
2. [Pytorch](https://pytorch.org/) >= 1.10
3. [Numpy](https://numpy.org/) >= 1.19
4. [SciPy](https://numpy.org/) >= 1.5.2
5. [matplotlib-base](https://matplotlib.org/) >= 3.3.1
6. [h5py](https://www.h5py.org/) >= 3.3.0
7. [torchinfo](https://pypi.org/project/torchinfo/) >= 0.1.5
8. [psutil](https://psutil.readthedocs.io/en/latest/) >= 5.8.0

It is <ins>strongly recommended</ins> to follow the guideline in the next section to install these dependencies.

---
## Installation

To ease the installation of SCPMel, two yaml files are provided to build a remote conda environnement that will <ins>automatically install the required dependencies</ins>. Once the conda environnement has been built, SCPMel can simply be installed with **pip**. These two steps are detailed hereafter.

Note that each yaml file will also install the following additional modules for easier integration of the remote environnement with Jupyter notebooks and Spyder.

- [nb_conda_kernels](https://github.com/Anaconda-Platform/nb_conda_kernels)
- [spyder-kernels](https://github.com/spyder-ide/spyder-kernels)
<br> </br>

### Building the conda environnement

Prior to build the remote conda environnement, a conda distribution needs to be installed; for further details see the documentation at: [https://www.anaconda.com](https://www.anaconda.com).

Once conda has been made available, the remote conda environnement is built through the following steps:

1. clone the repository: `git clone https://github.com/Gilquin/scpmel.git`
2. create the env: `$ conda env create -f [path_to_yaml]/[yaml_file].yml -p [path_to_env]/scpmel`
3. append it to your local list: `$ conda config --append envs_dirs [path_to_env]`

where: 

- `[path_to_yaml]` is the absolute path to the yaml file,
- `[yaml_file]` is the name of one of the two yaml file provided,
- `[path_to_env]` is the absolute path where the remote conda environnement will be installed, which is a user choice.

The choice of the yaml file depends on the hardware at the user disposal:

- *scpmel_gpu.yml* will build Pytorch with GPU support,
- *scpmel_cpu.yml* will build Pytorch only with CPU support.

To check that the remote conda environnement is correctly installed, run:
`$ conda activate scpmel`
<br> </br>

### Installing SCPMel

Once the remote environnement has been activated, the SCPMel package is installed by running the command:

`$ pip install . ` 

at the root of the cloned repository.

---
## References

```
@incollection{Pytorch,
	title = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
	author = {A. Paszke and S. Gross and M. Francisco and A. Lerer and J. Bradbury and G. Chanan and T. Killeen and Z. Lin and N. Gimelshein and L. Antiga and A. Desmaison and A. Kopf and E. Yang and Z. DeVito and M. Raison and A. Tejani and S. Chilamkurthyand B. Steiner and L. Fang and J. Bai and S. Chintala},
	booktitle = {Advances in Neural Information Processing Systems 32},
	editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
	pages = {8024--8035},
	year = {2019},
	publisher = {Curran Associates, Inc.}
}

@book{Hairer1993,
	title={Solving ordinary differential equations. 1, Nonstiff problems. Second Revised Edition},
	author={Hairer, Ernst and N{\o}rsett, Syvert P and Wanner, Gerhard},
	year={1993},
	publisher = {Springer-Verlag},
	address = {Berlin, Heidelberg},
	doi={10.1007/978-3-540-78862-1}
}

@book{Sagaut2006,
	author = {P. Sagaut and Y.-T. Lee},
	year = {2006},
	title = {Large Eddy Simulation for Incompressible Flows: An Introduction. Third Edition},
	publisher = {Springer-Verlag},
	address = {Berlin, Heidelberg},
	doi = {10.1007/b137536}
}

@misc{Zahm2021,
	title={Certified dimension reduction in nonlinear Bayesian inverse problems}, 
	author={O. Zahm and T. Cui and K. Law and A. Spantini and Y. Marzouk},
	year={2021},
	eprint={1807.03712},
	archivePrefix={arXiv},
	primaryClass={math.PR}
}
```
