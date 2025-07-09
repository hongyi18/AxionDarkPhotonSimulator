# AxionDarkPhotonSimulator
AxionDarkPhotonSimular is a python package for simulating an axion-dark photon system, coupled through a Chern-Simons term, in an expanding universe. Here, "axion" stands for a scalar field and "dark photon" stands for a massive photon field.

## Installation
One should download the folder with the version number, which includes all necessary python scripts. The main script for running the simulation is "main.py". For example, one can first navigate to the folder and then type `python main.py` in the command line.

## Documentation
This package was first released along with the research paper [arXiv:2507.](...). Mathematical formulations and numerical implementation details are provided in [Numerical details for AxionDarkPhotonSimulator.pdf](...). For more details about what this package can do or how to use it, simply read and play with the codes. Python is known for its readability, and the variables and functions in this package are clearly documented.

Prerequisite python version and modules: python 3.12, numpy 1.26.4, matplotlib 3.10.0, numba 0.61.0 (These versions have been tested).

## Code structure
The package includes the following scripts:
- *param.py*: Simulation settings and model-dependent quantities. Usually this is the only script that needs to be adjusted.
- *var.py*: Global variable buffers and functions used to manipulate them. It includes generic functions that other scripts need to call.
- *evol.py*: Functions used to evolve the system. It depends on *param.py* and *var.py*.
- *init.py*: Functions used to initialize the system. It depends on *param.py* and *var.py*.
- *output.py*: Functions used to calculate useful quantities, save data and make plots. It depends on *param.py* and *var.py*.
- *main.py*: The main script used to run the simulation. It depends on all other scripts.

## Key features
- Achieve 2nd-order accuracy in both time and space.
- Use implicit methods to ensure strong stability.
- Use the python module *numba* to accelerate the codes. On my personal laptop with Intel Core Ultra 7 155H, it takes ~20 hours to evolve the system with $128^3$ points by 8500 time steps.
- Estimate the total running time at the beginning of a simulation. You don't need to worry about that the simulation may take much longer time than you naively expect.
- Support resuming simulations from a breakpoint. The process is automated and you don't need to manipulate any files.
- Well structured and easy to maintain. The code blocks for different uses are well documented and separated.
- Supports different expanding backgrounds. One can specify a radiation- or matter-dominated background, or one with transitions between them. (In the current setup, the expansion of the universe is not dynamically driven by the matter fields, but is instead treated as a background.)

## Contact
If you find any bugs, please report them to my email address displayed on [my website](https://hongyi18.github.io/).
