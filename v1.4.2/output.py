# -*- coding: utf-8 -*-
"""
Title: AxionDarkPhotonSimulator
Author: Hong-Yi Zhang
Website: hongyi18.github.io
Version: 1.4.2
Date: May 3, 2025

Description: Functions used to calculate useful quantities, save data and make plots.
"""

from param import alpha, beta, N
from var import pd, a, H, take, pot_U, laplacian, divergenceX, curlX
from var import dir_data, save_list_data, load_list_data, plot_list_data, plot_grid_data
import var
from numba import njit, int32, float64
import numpy as np

#==============================================================================
# Calculate useful quantities.
# In these functions idx[0]=1 is assumed.
# t0 stands for current time at idx[0]=1.

@njit(float64(float64[:,:,:,:], float64[:,:,:,:,:], float64[:,:,:,:,:], int32[:], float64))
def cal_delta(phi, X, P, idx, t0):
    term1 = divergenceX(P, idx)
    term2 = laplacian(X[0], idx)
    term3 = (a(t0)*beta)**2*take(X[0], idx)
    term4 = 0
    for i in range(1, 4):
        term4 += pd(phi, idx, i)*curlX(X, idx, i)
    term4 *= alpha/a(t0)
    norm = abs(term1) + abs(term2) + abs(term3) + abs(term4)
    if norm!=0:
        return (term1 - term2 + term3 - term4)/norm
    else:
        return 0

@njit((float64[:,:,:,:], float64[:,:,:,:,:], float64[:,:,:,:,:], float64[:,:,:], float64[:]))
def update_delta(phi, X, P, delta, t):
    idx = np.ones(4, dtype='int32')
    for nx in range(N):
        idx[1] = nx
        for ny in range(N):
            idx[2] = ny
            for nz in range(N):
                idx[3] = nz
                delta[nx,ny,nz] = cal_delta(phi, X, P, idx, t[0])

@njit(float64(float64[:,:,:,:], float64[:,:,:,:], int32[:], float64))
def cal_rho_phi(phi, pi, idx, t0):
    rho = (take(pi, idx) - H(t0)*take(phi, idx))**2/2
    for i in range(1, 4):
        rho += pd(phi, idx, i)**2/2
    rho = rho/a(t0)**4 + pot_U(phi, idx, t0)
    return rho

@njit((float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:], float64[:]))
def update_rho_phi(phi, pi, rho_phi, t):
    idx = np.ones(4, dtype='int32')
    for nx in range(N):
        idx[1] = nx
        for ny in range(N):
            idx[2] = ny
            for nz in range(N):
                idx[3] = nz
                rho_phi[idx[1],idx[2],idx[3]] = cal_rho_phi(phi, pi, idx, t[0])

@njit(float64(float64[:,:,:,:,:], float64[:,:,:,:,:], int32[:], float64))
def cal_rho_X(X, P, idx, t0):
    term1 = 0
    for i in range(1, 4):
        term1 += (take(P[i], idx) - pd(X[0], idx, i))**2 + curlX(X, idx, i)**2
    term2 = beta**2/2/a(t0)**2*(take(X[0], idx)**2+take(X[1], idx)**2+take(X[2], idx)**2+take(X[3], idx)**2)
    return term1/2/a(t0)**4 + term2

@njit((float64[:,:,:,:,:], float64[:,:,:,:,:], float64[:,:,:], float64[:]))
def update_rho_X(X, P, rho_X, t):
    idx = np.ones(4, dtype='int32')
    for nx in range(N):
        idx[1] = nx
        for ny in range(N):
            idx[2] = ny
            for nz in range(N):
                idx[3] = nz
                rho_X[idx[1],idx[2],idx[3]] = cal_rho_X(X, P, idx, t[0]) 

@njit((float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:,:], float64[:,:,:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:]))
def update(phi, pi, X, P, delta, rho_phi, rho_X, t):
    update_delta(phi, X, P, delta, t)
    update_rho_phi(phi, pi, rho_phi, t)
    update_rho_X(X, P, rho_X, t)
    

#==============================================================================
# Save data.

def save_data():
    t0 = var.t[0]
    save_list_data(var.t_plot, t0, 't_plot')
    save_list_data(var.delta_plot, np.sqrt(np.mean(var.delta**2)), 'delta_plot')
    save_list_data(var.rhobar_phi_plot, np.mean(var.rho_phi), 'rhobar_phi_plot')
    save_list_data(var.rhobar_X_plot, np.mean(var.rho_X), 'rhobar_X_plot')
    save_list_data(var.rhomax_phi_plot, np.max(var.rho_phi), 'rhomax_phi_plot')
    save_list_data(var.rhomax_X_plot, np.max(var.rho_X), 'rhomax_X_plot')
    save_list_data(var.phibar_plot, np.mean(var.phi[1]), 'phibar_plot')
    save_list_data(var.phimax_plot, np.max(var.phi[1]), 'phimax_plot')

def save_checkpoint():
    n = var.detect_checkpoint_number()
    with open(dir_data + 'checkpoint_time.dat', 'a') as f:
        f.write('%s\n' % var.t[0])
    var.save_grid_data(var.phi[1], 'checkpoint_phi_'+str(n))
    var.save_grid_data(var.pi[1], 'checkpoint_pi_'+str(n))
    for mu in range(4):
        var.save_grid_data(var.X[mu,1], 'checkpoint_X'+str(mu)+'_'+str(n))
        var.save_grid_data(var.P[mu,1], 'checkpoint_P'+str(mu)+'_'+str(n))

# Check if data lists contain the expected number of elements.
def check_data_integrity():
    for n in range(len(var.plot_list)):
        datalist = load_list_data(var.plot_list[n])
        if eval('len(var.%s) != len(datalist)' % var.plot_list[n]):
            np.savetxt(dir_data + var.plot_list[n]+'.dat', datalist)
            var.printf('(%s.dat is corrupted and fixed.)' % var.plot_list[n])
    

#==============================================================================
# Make plots.

def plot_data():
    plot_list_data(var.t_plot, var.delta_plot, 't_delta', 'y')
    plot_list_data(var.t_plot, var.rhobar_phi_plot, 't_rhobar_phi', 'y')
    plot_list_data(var.t_plot, var.rhobar_X_plot, 't_rhobar_X', 'y')
    plot_list_data(var.t_plot, np.array(var.rhobar_phi_plot) + np.array(var.rhobar_X_plot), 't_rhobar_all', 'y')
    plot_list_data(var.t_plot, var.rhomax_phi_plot, 't_rhomax_phi', 'y')
    plot_list_data(var.t_plot, var.rhomax_X_plot, 't_rhomax_X', 'y')
    plot_list_data(var.t_plot, var.phibar_plot, 't_phibar', 'none')
    plot_list_data(var.t_plot, var.phimax_plot, 't_phimax', 'none')

def plot_grid(n):
    plot_grid_data(var.phi[1], 'snapshot_phi_'+str(n), 'mean')
    plot_grid_data(var.pi[1], 'snapshot_pi_'+str(n), 'mean')
    for mu in range(4):
        plot_grid_data(var.X[mu,1], 'snapshot_X'+str(mu)+'_'+str(n), 'mean')
        plot_grid_data(var.P[mu,1], 'snapshot_P'+str(mu)+'_'+str(n), 'mean')
    plot_grid_data(var.delta, 'snapshot_delta_'+str(n), 'rms')
    plot_grid_data(var.rho_phi, 'snapshot_rho_phi_'+str(n), 'mean')
    plot_grid_data(var.rho_X, 'snapshot_rho_X_'+str(n), 'mean')

def plot():
    n = var.detect_snapshot_number()
    with open(dir_data + 'snapshot_time.dat', 'a') as f:
        f.write('%s\n' % var.t[0])
    plot_data()
    plot_grid(n)