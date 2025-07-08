# -*- coding: utf-8 -*-
r"""
Title: AxionDarkPhotonSimulator
Author: Hong-Yi Zhang
Website: hongyi18.github.io
Version: 1.4.2
Date: May 3, 2025

Description: Functions used to evolve the system.
"""

from param import N, alpha, beta, dt_coef, dx, m2f, lamb_s
from var import pd, cycle, cycle2, a, H, laplacian, curlX, pot_Ud, take
from numba import njit, int32, float64
import numpy as np

#==============================================================================
# 1. Evolve \phi and X_\mu by half a step with the forward difference method.
# Save the intermediate values in phi[2] and X[mu,2].

@njit((float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:,:], float64[:,:,:,:,:], float64))
def evolve_phi_Xmu_half1(phi, pi, X, P, dt):
    for nx in range(N):
        for ny in range(N):
            for nz in range(N):
                phi[2,nx,ny,nz] = phi[1,nx,ny,nz] + pi[1,nx,ny,nz]*dt/2 
                for mu in range(4):
                    X[mu,2,nx,ny,nz] = X[mu,1,nx,ny,nz] + P[mu,1,nx,ny,nz]*dt/2


#==============================================================================
# 2. Evolve \pi and P_i by solving implicit matrix equations.
# In these functions idx[0]=2 is assumed by default.
# Fields at idx[0]=2 are those at the middle step, e.g., m+1/2.
# t0 stands for time at the middle step, e.g., m+1/2.

@njit(float64(float64[:,:,:,:,:], int32[:], float64, float64, int32))
def cal_b(X, idx, t0, dt, i):
    return dt/2*alpha*((beta*m2f)**2/lamb_s)/a(t0) * curlX(X, idx, i)
    
@njit(float64(float64[:,:,:,:], int32[:], float64, float64, int32))
def cal_c(phi, idx, t0, dt, i):
    return dt/2*alpha/a(t0)*pd(phi, idx, i)

@njit(float64(float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:,:], float64[:,:,:,:,:], int32[:], float64, float64))
def cal_d0(phi, pi, X, P, idx, t0, dt):
    d0 = pi[1,idx[1],idx[2],idx[3]] + dt*(laplacian(phi, idx) - a(t0)**3*pot_Ud(phi, idx, t0)) # a''/a=0
    for i in range(1, 4):
        d0 -= cal_b(X, idx, t0, dt, i)*(P[i,1,idx[1],idx[2],idx[3]] - 2*pd(X[0], idx, i))
    return d0

@njit(float64(float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:,:], float64[:,:,:,:,:], int32[:], float64, float64, int32))
def cal_di(phi, pi, X, P, idx, t0, dt, i):
    di = P[i,1,idx[1],idx[2],idx[3]]
    di += dt*(laplacian(X[i], idx) - 2*H(t0)*pd(X[0], idx, i) - (a(t0)*beta)**2*take(X[i], idx))
    di += cal_b(X, idx, t0, dt, i)*(pi[1,idx[1],idx[2],idx[3]] - 2*H(t0)*take(phi, idx))
    di += (P[cycle(i),1,idx[1],idx[2],idx[3]] - 2*pd(X[0], idx, cycle(i)))*cal_c(phi, idx, t0, dt, cycle2(i))
    di -= (P[cycle2(i),1,idx[1],idx[2],idx[3]] - 2*pd(X[0], idx, cycle2(i)))*cal_c(phi, idx, t0, dt, cycle(i))
    return di

# ::1 is used in numba signatures to indicate contiguous arrays.
@njit((float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:,:], float64[:,:,:,:,:], int32[:], float64, float64, float64[:,::1], float64[::1]))
def cal_pi_Pi(phi, pi, X, P, idx, t0, dt, bc_mat, d_vec):
    d_vec[0] = cal_d0(phi, pi, X, P, idx, t0, dt)
    for i in range(1, 4):
        d_vec[i] = cal_di(phi, pi, X, P, idx, t0, dt, i)
        bc_mat[0,i] = cal_b(X, idx, t0, dt, i)
        bc_mat[i,0] = -bc_mat[0,i]
        for j in range(1, 4):
            if j>i:
                bc_mat[i,j] = (-1)**(i+j)*cal_c(phi, idx, t0, dt, 6-i-j)
            elif j<i:
                bc_mat[i,j] = -bc_mat[j,i]
    
    temp = np.dot(np.linalg.inv(bc_mat), d_vec) # np.dot asks for gontiguous arrays.
    pi[2,idx[1],idx[2],idx[3]] = temp[0]
    for i in range(1, 4):
        P[i,2,idx[1],idx[2],idx[3]] = temp[i]

@njit((float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:,:], float64[:,:,:,:,:], float64[:]))
def evolve_pi_Pi(phi, pi, X, P, t):
    bc_mat = np.identity(4, dtype='float64') # Implicit equations for pi and P.
    d_vec = np.zeros(4, dtype='float64') # Implicit equations for pi and P.
    idx = np.zeros(4, dtype='int32')
    idx[0] = 2
    for nx in range(N):
        idx[1] = nx
        for ny in range(N):
            idx[2] = ny
            for nz in range(N):
                idx[3] = nz
                cal_pi_Pi(phi, pi, X, P, idx, t[0]+t[1]/2, t[1], bc_mat, d_vec)


#==============================================================================
# 3. Evolve \phi and X_i by another half step with the backward method.
@njit((float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:,:], float64[:,:,:,:,:], float64))
def evolve_phi_Xi_half2(phi, pi, X, P, dt):
    for nx in range(N):
        for ny in range(N):
            for nz in range(N):
                phi[2,nx,ny,nz] += pi[2,nx,ny,nz]*dt/2 
                for i in range(1, 4):
                    X[i,2,nx,ny,nz] += P[i,2,nx,ny,nz]*dt/2
    return 0


#==============================================================================
# 4. Evolve X_0 and P_0 by solving the equation with Newton's method.

@njit((float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:,:], float64[:,:,:,:,:], int32[:], float64, float64))
def cal_X0_P0(phi, pi, X, P, idx, t0, dt):
    X0_old = take(X[0], idx)
    term = 0
    for i in range(1, 4):
        term += pd(X[i], idx, i)
    X[0,2,idx[1],idx[2],idx[3]] = (X0_old + term*dt/2)/(1+dt*H(t0))
    P[0,2,idx[1],idx[2],idx[3]] = term - 2*H(t0)*take(X[0], idx)

@njit((float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:,:], float64[:,:,:,:,:], float64[:]))
def evolve_X0_P0(phi, pi, X, P, t):
    idx = np.zeros(4, dtype='int32')
    idx[0] = 2
    for nx in range(N):
        idx[1] = nx
        for ny in range(N):
            idx[2] = ny
            for nz in range(N):
                idx[3] = nz
                cal_X0_P0(phi, pi, X, P, idx, t[0]+t[1], t[1])


#==============================================================================
# 5. Evolve all fields over the whole grid.

@njit()
def update_t(t):
    t[0] += t[1]
    t[1] = dt_coef*min(dx, 1/a(t[0]), 1/beta/a(t[0]))
    
@njit((float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:,:], float64[:,:,:,:,:], float64[:]))
def evolve(phi, pi, X, P, t):
    evolve_phi_Xmu_half1(phi, pi, X, P, t[1])
    evolve_pi_Pi(phi, pi, X, P, t)
    evolve_phi_Xi_half2(phi, pi, X, P, t[1])
    evolve_X0_P0(phi, pi, X, P, t)
    for nt in range(2):
        phi[nt,:,:,:] = phi[nt+1,:,:,:]
        pi[nt,:,:,:] = pi[nt+1,:,:,:]
        for mu in range(4):
            X[mu,nt,:,:,:] = X[mu,nt+1,:,:,:]
            P[mu,nt,:,:,:] = P[mu,nt+1,:,:,:]
    update_t(t)
    return 0