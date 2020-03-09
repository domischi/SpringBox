import numba
import numpy as np
from scipy.spatial.distance import pdist, squareform

@numba.njit(parallel=True)
def point_in_active_region(ps, AR):
    return (ps[:,0]>-1) * (ps[:,0]<1) * (ps[:,1]>-AR) * (ps[:,1]<AR)

@numba.jit
def RHS(particles, cutoff, lower_cutoff,k, AR, r0):
    rhs = np.zeros_like(particles)
    n_part=len(particles)
    acc = point_in_active_region(particles, AR)
    Dij = squareform(pdist(particles))
    Iij = Dij * np.outer(acc,acc)
    Iij = (Iij>lower_cutoff) * (Iij<cutoff)
    for i in range(n_part):
        for j in range(i+1,n_part):
            if Iij[i,j]!=0:
                rhs[i] += -k*((Dij[i,j]-r0)/Dij[i,j])*(particles[i]-particles[j])
                rhs[j] += +k*((Dij[i,j]-r0)/Dij[i,j])*(particles[i]-particles[j])
    return rhs

@numba.jit
def integrate_one_timestep(particles, velocities, dt, m , cutoff, lower_cutoff, k, AR, drag_factor, r0=r0):
    particles = particles + dt * velocities
    velocities = (1-drag_factor)*velocities + dt/m * RHS(particles,cutoff=cutoff, lower_cutoff=lower_cutoff,k=k, AR=AR)
    return particles, velocities
