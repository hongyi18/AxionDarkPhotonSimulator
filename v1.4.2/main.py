# -*- coding: utf-8 -*-
r"""
Title: AxionDarkPhotonSimulator
Author: Hong-Yi Zhang
Website: hongyi18.github.io
Version: 1.4.2
Date: May 3, 2025

Description: Main script. 

Related scripts include:
param.py    Simulation settings and model-dependent quantities
var.py      Global variable buffers and functions used to manipulate them.
evol.py     Functions used to evolve the system.
init.py     Functions used to initialize the system.  
output.py   Functions used to calculate useful quantities, save data and make plots.
"""

import time
t_start = time.time()
import var
import param
from var import phi, pi, X, P, t, delta, rho_phi, rho_X
import init
import evol
import output
from numba import njit
import os

# Check if data folder exists.
if not os.path.exists(var.dir_data):
    os.makedirs(var.dir_data)
var.printf('Loading modules is complete: %.0fs' % (time.time()-t_start))

def precheck():
    if param.Nt%param.Nt_save_data!=0:
        var.printf('(Warning: Nt/Nt_save_data is not an integer. Physical quantities at the final time will not be calculated or saved.)')
    if param.Nt/param.Nt_save_checkpoint<1:
        var.printf('(Warning: Nt/Nt_save_checkpoint is less than 1. No grid data will be saved.)')
    elif param.Nt%param.Nt_save_checkpoint!=0:
        var.printf('(Warning: Nt/Nt_save_checkpoint is not an integer. Grid data at the final time will not be saved.)')    
        
    # Clean files in the data folder.
    if param.if_auto_initialize_with_last_checkpoint==0:
        for f in os.listdir(var.dir_data):
            os.remove(os.path.join(var.dir_data, f))
        var.printf('Data folder is purged: %.0fs' % (time.time()-t_start))

@njit
def estimate_final_t(t):
    t_original = t.copy()
    for nt in range(param.Nt):
        evol.update_t(t)
    t_final = t.copy()
    t[:] = t_original[:] # Recover the original t.
    return t_final
    
def simulate():
    t_simulate = time.time()
    for nt in range(param.Nt):
        evol.evolve(phi, pi, X, P, t)
    
        if (nt+1)%param.Nt_save_data==0:
            output.update(phi, pi, X, P, delta, rho_phi, rho_X, t)
            output.save_data()
        
        if (nt+1)%param.Nt_plot==0:
            output.update(phi, pi, X, P, delta, rho_phi, rho_X, t)
            output.plot()
                
        if (nt+1)%param.Nt_save_checkpoint==0:
            output.save_checkpoint()
    
        if (nt+1)%param.Nt_prog==0:
            num_prog = (nt+1)/param.Nt_prog
            num_prog_total = param.Nt//param.Nt_prog
            var.printf('Progress %.0f%%: %.0fs' % (100*num_prog/num_prog_total, time.time()-t_start))
        
        # Predict the time needed to complete the simulation.
        if t_simulate!=0:
            t_past = time.time()-t_simulate
            if t_past>10:
                t_rest = t_past*(param.Nt/(nt+1) - 1) 
                var.printf('(Simulation is expected to complete after %.0fs.)' % t_rest)
                t_simulate = 0


if __name__=='__main__':        
    precheck()
    init.initialize()
    output.update(phi, pi, X, P, delta, rho_phi, rho_X, t)
    output.plot()
    if param.if_save_initial_checkpoint==1:
        output.save_checkpoint()
    t_final = estimate_final_t(t)
    var.printf('Initialization is complete: %.0fs' % (time.time()-t_start))
    var.printf('(Simulation will be completed at numerical time %f with time step %f.)' % (t_final[0], t_final[1]))
    
    var.printf('Simulation is started: %.0fs' % (time.time()-t_start))
    simulate()
    if param.Nt>=param.Nt_save_data:
        output.check_data_integrity()
    var.printf('Simulation is complete: %.0fs' % (time.time()-t_start))