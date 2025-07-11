# -*- coding: utf-8 -*-
r"""
Title: AxionDarkPhotonSimulator
Author: Hong-Yi Zhang
Website: hongyi18.github.io
Version: 1.4.2
Date: May 3, 2025

Description: Simulation settings and model-dependent quantities.
"""

import numpy as np

# Lattice settings.
L = np.pi/2
N = 32
dx = L/N

Nt = 1000 # Total number of time steps.
dt_coef = 0.25 # dt = dx * dt_coef


# Settings for the cosmological background.
which_expansion_scheme = 'RD'
# 'none': no expansion.
# 'RD': radiation-dominated background.
# 'MD': matter-dominated background.
# 'RD->MD': transition from a radiation- to matter-dominated background.
# 'MD->RD': transition from a matter- to radiation-dominated background.

ti = 0.1 # Initial conformal time.
t_transition = ti # The conformal time when a transition between backgrounds occurs, used for 'RD->MD' or 'MD->RD'.


# Settings for initial conditions.
if_auto_initialize_with_last_checkpoint = 0 
# 1: If there is a breakpoint, resume the simulation automatically.
# 0: Start a new simulation every time, even if there is a breakpoint.

if_save_initial_checkpoint = 0
# 1: Save the initial grid data for all fields.
# 0: Don't save the initial grid data.

which_initial_condition = 'vacuum fluctuations' 
# 'custom': I want to specify custom initial conditions in "init.py".
# 'vacuum fluctuations': Intialize small perturbations according to vacuum fluctuations. See the pdf documentation for details.
# 'data files': Load initial conditions from data files in the folder "initial/" (create it yourself). Read the documentation for "initialize_data_file()" in "init.py".

phi_initial = 5 # The initial misalignment value of the axion field, used for 'vacuum fluctuations'.


# Settings for simulation outputs.
Nt_plot = max(1, Nt//20) # Plot current data lists every Nt_plot time steps.
Nt_save_checkpoint = max(1, Nt//1) # Save the grid data, i.e. N*N*N matrices, for all fields every Nt_save_checkpoint time steps. Too many checkpoint files could occupy a lot of disk storage.
Nt_save_data = max(1, Nt//500) # Save data lists every Nt_save_data time steps.
Nt_prog = max(1, Nt//50) # Report the running progress every Nt_prog time steps.


# Model parameters.
alpha = 0 # Coupling constant.
beta = 0.1 # m_X/m_\phi.
m2f = 1e-5 # m_\phi/f_\phi.
lamb_s = (m2f*beta)**2 # Rescaling factor.
