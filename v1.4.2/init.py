# -*- coding: utf-8 -*-
r"""
Title: AxionDarkPhotonSimulator
Author: Hong-Yi Zhang
Website: hongyi18.github.io
Version: 1.4.2
Date: May 3, 2025

Description: Functions used to initialize the system.    
"""

from param import L, N, dx, beta, ti, lamb_s, m2f, dt_coef, if_auto_initialize_with_last_checkpoint, which_initial_condition, phi_initial
from var import a, H, take, pd
from var import detect_checkpoint_number, load_grid_data, append_list_data, load_list_data
from var import phi, pi, X, P, t
import var
from numba import njit, int32, float64
import numpy as np

co_k = 2*np.pi*np.fft.fftfreq(N, dx) # Fourier coordinates.
kx, ky, kz = np.meshgrid(co_k, co_k, co_k, indexing='ij')

#==============================================================================
# Custom initial conditions.

def custom_phi(phi, pi, X, P):
    pass

def custom_pi(phi, pi, X, P):
    pass

def custom_Xi(phi, pi, X, P):
    r0 = L/4
    co = np.arange(0, L, L/N)
    x, y, z = np.meshgrid(co, co, co, indexing='ij')
    rsq = (x-L/2)**2 + (y-L/2)**2 + (z-L/2)**2
    X[1,1,:,:,:] = np.sqrt(lamb_s/beta**2)*np.exp(-rsq/r0**2)
        
def custom_Pi(phi, pi, X, P):
    pass
        

#==============================================================================
# Initialize dark photon fields with vacuum fluctuations.

def vacuum_fluc_phi(phi):
    m = a(t[0])
    omega = np.sqrt(kx**2+ky**2+kz**2+m**2)
    std = np.sqrt(m2f**2/(4*omega))
    real = np.random.normal(0, 1, size=std.shape)*std
    imag = np.random.normal(0, 1, size=std.shape)*std
    delta_phi = np.real(np.fft.ifftn((real+1j*imag)*np.sqrt(L**3)/dx**3))
    phi[1,:,:,:] = phi_initial*a(t[0]) + delta_phi

def vacuum_fluc_pi(phi, pi):
    m = a(t[0])
    omega = np.sqrt(kx**2+ky**2+kz**2+m**2)
    std = np.sqrt(m2f**2*omega/4)
    real = np.random.normal(0, 1, size=std.shape)*std
    imag = np.random.normal(0, 1, size=std.shape)*std
    delta_pi = np.real(np.fft.ifftn((real+1j*imag)*np.sqrt(L**3)/dx**3))
    pi[1,:,:,:] = H(t[0])*phi_initial*a(t[0]) + delta_pi

def vacuum_fluc_Xi(X):
    m = a(t[0])*beta
    omega = np.sqrt(kx**2+ky**2+kz**2+m**2)
    std = np.sqrt(lamb_s/(4*omega*beta**2))
    for i in range(1, 4):
        real = np.random.normal(0, 1, size=std.shape)*std
        imag = np.random.normal(0, 1, size=std.shape)*std
        X[i,1,:,:,:] = np.real(np.fft.ifftn((real+1j*imag)*np.sqrt(L**3)/dx**3))

def vacuum_fluc_Pi(P):
    m = a(t[0])*beta
    omega = np.sqrt(kx**2+ky**2+kz**2+m**2)
    std = np.sqrt(lamb_s*omega/(4*beta**2))
    for i in range(1, 4):
        real = np.random.normal(0, 1, size=std.shape)*std
        imag = np.random.normal(0, 1, size=std.shape)*std
        P[i,1,:,:,:] = np.real(np.fft.ifftn((real+1j*imag)*np.sqrt(L**3)/dx**3))
    

#==============================================================================
# Given Xi and Pi, solve X0 and P0.

def update_X0(X, P):
    m = a(t[0])*beta
    kx, ky, kz = np.meshgrid(co_k, co_k, co_k, indexing='ij')
    omega_sq = kx**2+ky**2+kz**2+m**2
    # If N is even, the Fourier mode at index N//2 represents both the positive and negative Nyquist frequencies.
    # Now we calculate k*Pk, we can define co_k[N//2]=0 such that the inverse transformation yields real-valued functions.
    if N%2==0:
        kx[N//2,:,:] = 0
        ky[:,N//2,:] = 0
        kz[:,:,N//2] = 0
    term = kx*np.fft.fftn(P[1,1]) + ky*np.fft.fftn(P[2,1]) + kz*np.fft.fftn(P[3,1])
    X0 = np.fft.ifftn( -1j*term/omega_sq )
    
    # The imaginary part of X0 should be 0 within the machine precision.
    if np.any(np.abs(np.imag(X0))>np.finfo(float).eps):
        raise ValueError('initialize_X0() outputs imaginary values.')
    else:
        X[0,1,:,:,:] = np.real(X0)

@njit((float64[:,:,:,:,:], float64[:,:,:,:,:], int32[:]))
def cal_P0(X, P, idx):
    term = 0
    for i in range(1, 4):
        term += pd(X[i], idx, i)
    return term - 2*H(t[0])*take(X[0], idx)

@njit((float64[:,:,:,:,:], float64[:,:,:,:,:]))
def update_P0(X, P):
    idx = np.ones(4, dtype='int32')
    for nx in range(N):
        idx[1] = nx
        for ny in range(N):
            idx[2] = ny
            for nz in range(N):
                idx[3] = nz
                P[0,1,nx,ny,nz] = cal_P0(X, P, idx)

def update_X0_P0(X, P):
    update_X0(X, P)
    update_P0(X, P)
    
    
#==============================================================================
# Initialization.

def initialize_custom():
    t[0] = ti
    t[1] = dt_coef*min(dx, 1/a(t[0]), 1/beta/a(t[0]))
    
    custom_phi(phi, pi, X, P)
    custom_pi(phi, pi, X, P)
    custom_Xi(phi, pi, X, P)
    custom_Pi(phi, pi, X, P)
    update_X0_P0(X, P)
    
def initialize_vaccum_fluc():
    t[0] = ti
    t[1] = dt_coef*min(dx, 1/a(t[0]), 1/beta/a(t[0]))
    
    vacuum_fluc_phi(phi)
    vacuum_fluc_pi(phi, pi)
    vacuum_fluc_Xi(X)
    vacuum_fluc_Pi(P)
    update_X0_P0(X, P)
    
# Initialize with the last checkpoint data.
def initialize_checkpoint():
    checkpoint_time = load_list_data('checkpoint_time')
    t[0] = checkpoint_time[-1]
    t[1] = dt_coef*min(dx, 1/a(t[0]), 1/beta/a(t[0]))
    
    n = detect_checkpoint_number()
    phi[1] = load_grid_data('checkpoint_phi_'+str(n-1))
    pi[1] = load_grid_data('checkpoint_pi_'+str(n-1))
    for mu in range(4):
        X[mu,1] = load_grid_data('checkpoint_X'+str(mu)+'_'+str(n-1))
        P[mu,1] = load_grid_data('checkpoint_P'+str(mu)+'_'+str(n-1))
    
    # Load data lists.
    for n in range(len(var.plot_list)):
        append_list_data(var.plot_list[n], eval('var.'+var.plot_list[n]))

# Initialize with local data files.
# One should prepare data files, which save elements of N*N*N matrices, for all fields in a subfolder named "initial" (create it yourself).
# The data file should be named with "field name" + ".dat", such as "phi.dat", "X0.dat".
# To prepare data files, one may directly copy and rename the checkpoint data file from a previous simulation.
# This feature could be useful if one would like to run several different simulations starting from identical initial conditions.
def initialize_data_file():
    temp = var.dir_data
    var.dir_data = 'initial/' # Change the directory for file io functions.
    
    t[0] = ti
    t[1] = dt_coef*min(dx, 1/a(t[0]), 1/beta/a(t[0]))
    
    phi[1] = load_grid_data('phi') # Load the data from "initial/phi.dat" as initial conditions for phi field.
    pi[1] = load_grid_data('pi')
    for mu in range(4):
        X[mu,1] = load_grid_data('X'+str(mu))
        P[mu,1] = load_grid_data('P'+str(mu))
    var.dir_data = temp # Recover the original directory.

def initialize():
    if if_auto_initialize_with_last_checkpoint==1 and detect_checkpoint_number()>0:
        initialize_checkpoint()
    elif which_initial_condition=='custom':
        initialize_custom()
    elif which_initial_condition=='vacuum fluctuations':
        initialize_vaccum_fluc()
    elif which_initial_condition=='data files':
        initialize_data_file()
    else:
        raise ValueError('Invalid input for which_initial_condition.')