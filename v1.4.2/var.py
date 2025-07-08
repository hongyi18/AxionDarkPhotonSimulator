# -*- coding: utf-8 -*-
r"""
Title: AxionDarkPhotonSimulator
Author: Hong-Yi Zhang
Website: hongyi18.github.io
Version: 1.4.2
Date: May 3, 2025

Description: Global variable buffers and functions used to manipulate them.
"""

from param import N, ti, which_expansion_scheme, t_transition, dt_coef, dx, L
from numba import njit, int32, float64
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

#==============================================================================
# Spacetime background.

# Scale factor in conformal time.
@njit(float64(float64))
def a(t):
    if which_expansion_scheme=='RD->MD': # Radiation to matter domination.
        a_transition = 2/( (ti/t_transition)**2 + (ti/t_transition) )
        return a_transition/2 * ( (t/t_transition)**2 + (t/t_transition) )
    elif which_expansion_scheme=='MD->RD': # Matter to radiation domination.
        a_transition = ( (t_transition/ti)**2 + (t_transition/ti) )/2
        return 2*a_transition / ( (t_transition/t)**2 + (t_transition/t) )
    # For a single type of the background, choose the equation of state parameter.
    else:
        if which_expansion_scheme=='none': # No expansion.
            w = np.inf 
        elif which_expansion_scheme=='RD': # Radiation domination.
            w = 1/3
        elif which_expansion_scheme=='MD': # Matter domination.
            w = 0
        return (t/ti)**(2/(1+3*w))

# Conformal Hubble parameter in conformal time.
@njit(float64(float64))
def H(t):
    return (a(t+1e-8) - a(t-1e-8))/2e-8 / a(t)

    
#==============================================================================
# Fields.

# Buffers.
phi = np.zeros((3, N, N, N), dtype='float64') # The indices refer to (time, x-axis, y-axis, z-axis).
pi = np.zeros((3, N, N, N), dtype='float64')
X = np.zeros((4, 3, N, N, N), dtype='float64') # The indices refer to (mu, time, x-axis, y-axis, z-axis)
P = np.zeros((4, 3, N, N, N), dtype='float64')

# Useful quantities.
t = np.array([ti, dt_coef*dx]) # Current (conformal) time and time step.
delta = np.zeros((N, N, N), dtype='float64') # Monitor numerical accuracy.
rho_phi = np.zeros((N, N, N), dtype='float64') # Energy density of axions.
rho_X = np.zeros((N, N, N), dtype='float64') # Energy density of dark photons.

# Data list to be calculated and saved. If new lists are added, also modify the relevant codes in output.py.
plot_list = ['t_plot', 'delta_plot', 'rhobar_phi_plot', 'rhobar_X_plot', \
             'rhomax_phi_plot', 'rhomax_X_plot', 'phibar_plot', 'phimax_plot'] # Name of lists.
for n in range(len(plot_list)):
    exec(plot_list[n]+'=[]') # Create empty list for names listed in plot_list.

@njit(float64(float64[:,:,:,:], int32[:]))
def take(field, idx):
    return field[idx[0],idx[1],idx[2],idx[3]]


#==============================================================================
# Field potentials.

@njit(float64(float64[:,:,:,:], int32[:], float64))
def pot_U(phi, idx, t):
    original_phi = take(phi, idx)/a(t)
    return original_phi**2/(2+original_phi**2)

@njit(float64(float64[:,:,:,:], int32[:], float64))
def pot_Ud(phi, idx, t):
    original_phi = take(phi, idx)/a(t)
    return 4*original_phi/(2+original_phi**2)**2
        

#==============================================================================
# Functions used to manipulate variables.

# Derivatives in a periodic box.
@njit((int32[:], int32))
def incre(idx, mu):
    if mu==0:
        idx[mu] += 1 
    else:
        if idx[mu]==N-1:
            idx[mu] = 0 
        else:
            idx[mu] += 1

@njit((int32[:], int32))
def incre2(idx, mu):
    if mu==0:
        idx[mu] += 2 
    else:
        if idx[mu]==N-2:
            idx[mu] = 0
        elif idx[mu]==N-1:
            idx[mu] = 1 
        else:
            idx[mu] += 2

@njit((int32[:], int32))
def decre(idx, mu):
    if mu==0:
        idx[mu] -= 1 
    else:
        if idx[mu]==0:
            idx[mu] = N-1 
        else:
            idx[mu] -= 1

@njit((int32[:], int32))
def decre2(idx, mu):
    if mu==0:
        idx[mu] -= 2 
    else:
        if idx[mu]==0:
            idx[mu] = N-2
        elif idx[mu]==1:
            idx[mu] = N-1 
        else:
            idx[mu] -= 2

# Partial derivatives.
@njit(float64(float64[:,:,:,:], int32[:], int32))
def pd(field, idx, mu):
    if mu in [1,2,3]:
        incre(idx, mu)
        field_next = take(field, idx)
        decre2(idx, mu)
        field_prev = take(field, idx)
        incre(idx, mu)  # Recover the original idx.
        return (field_next-field_prev)/(2*dx)
    else:
        raise ValueError('Invalid input for pd().')

# 2nd-order partial derivatives.
@njit(float64(float64[:,:,:,:], int32[:], int32, int32))
def pd2(field, idx, mu, nu):
    if (mu in [1,2,3]) and (nu in [1,2,3]):
        if mu==nu:
            incre(idx, mu)
            field_next = take(field, idx)
            decre2(idx, mu)
            field_prev = take(field, idx)
            incre(idx, mu)  # Recover the original idx.
            return (field_next-2*take(field, idx)+field_prev)/dx**2
        else:
            incre(idx, mu)
            field_next = pd(field, idx, nu)
            decre2(idx, mu)
            field_prev = pd(field, idx, nu)
            incre(idx, mu)  # Recover the original idx.
            return (field_next-field_prev)/(2*dx)
    else:
        raise ValueError('Invalid input for pd2().')

# Increase the cyclic space index.
@njit(int32(int32))
def cycle(i):
    if i==3:
        return 1 
    else:
        return i+1

@njit(int32(int32))
def cycle2(i):
    if i==2:
        return 1 
    elif i==3:
        return 2 
    else:
        return 3

# Derivative operators.
@njit(float64(float64[:,:,:,:], int32[:]))
def laplacian(field, idx):
    lap = 0
    for i in range(1, 4):
        lap = lap + pd2(field, idx, i, i)
    return lap

@njit(float64(float64[:,:,:,:,:], int32[:]))
def divergenceX(X, idx):
    div = 0
    for i in range(1, 4):
        div += pd(X[i], idx, i)
    return div

@njit(float64(float64[:,:,:,:,:], int32[:], int32))
def curlX(X, idx, i):
    return pd(X[cycle2(i)], idx, cycle(i)) - pd(X[cycle(i)], idx, cycle2(i))

#==============================================================================
# Functions used for file io.

dir_data = 'data/' # Directory to save and load data. Add "/" at the end.

# Save the printing message to a file.
def printf(x):
    print(x)
    with open(dir_data + 'simulation_message.txt', 'a') as f:
        print(x, file=f, flush=True)

# Append a data point to the corresponding list and save it to the data file.
def save_list_data(data_list, data_point, file_name):
    data_list.append(data_point)
    with open(dir_data + file_name + '.dat', 'a') as f:
        f.write('%s\n' % data_list[-1])

def load_list_data(file_name):
    return np.loadtxt(dir_data + file_name + '.dat', ndmin=1)

# Read a data list file and append the elements to the existing list.
# This is used for resuming a simulation from checkpoint data.
def append_list_data(file_name, data_list):
    temp = load_list_data(file_name)
    if temp.ndim!=1:
        raise ValueError('Invalid input for append_list_data')
    for n in range(len(temp)):
        data_list.append(temp[n])

def save_grid_data(field, file_name):
    np.savetxt(dir_data + file_name + '.dat', field.reshape(N**3))
    
def load_grid_data(file_name):
    return np.loadtxt(dir_data + file_name + '.dat').reshape((N,N,N))
   
# Detect the total number of checkpoints.
def detect_checkpoint_number():
    if any('checkpoint_time.dat' in file_name for file_name in os.listdir(dir_data)):
        checkpoint_time = load_list_data('checkpoint_time')
        return len(checkpoint_time)
    else:
        return 0

# Detect the total number of snapshots.
def detect_snapshot_number():
    if any('snapshot_time.dat' in file_name for file_name in os.listdir(dir_data)):
        snapshot_time = load_list_data('snapshot_time')
        return len(snapshot_time)
    else:
        return 0

warnings.filterwarnings('error') # Turn matching warnings into exceptions.
# This is added because plt.xscale('log') somestimes raises user warnings that cannot be captured by "try... except...".
def try_log_scale(axis):
    if axis=='x':
        try:
            plt.xscale('log')
        except:
            plt.xscale('linear')
    elif axis=='y':
        try:
            plt.yscale('log')
        except:
            plt.yscale('linear')
    elif axis=='both':
        try_log_scale('x')
        try_log_scale('y')
    elif axis=='none':
        pass
    else:
        raise ValueError('Invalid input for try_log_scale().')

def plot_list_data(x, y, file_name, log_axis):
    plt.figure()
    plt.plot(x, y)
    try_log_scale(log_axis)
    plt.savefig(dir_data + file_name + '.png', bbox_inches='tight')
    plt.close()

def plot_grid_data(x, file_name, proj_method):
    plt.figure()
    # The input x should be a 3D array.
    if proj_method=='mean':
        y = np.mean(x, axis=2)
    elif proj_method=='rms':
        y = np.sqrt(np.mean(x**2, axis=2))
    else:
        raise ValueError('Invalid input for plot_grid_data()')
    plt.imshow(y, origin='lower', extent=(0, L, 0, L))
    plt.colorbar()
    plt.savefig(dir_data + file_name + '.png', bbox_inches='tight')
    plt.close()