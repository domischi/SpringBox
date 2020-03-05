import numba
import numpy as np
from scipy.spatial.distance import pdist, squareform

@numba.njit(parallel=True)
def point_in_active_region(ps, AR):
    return (ps[:,0]>-1) * (ps[:,0]<1) * (ps[:,1]>-AR) * (ps[:,1]<AR)

@numba.jit
def RHS(particles, cutoff, lower_cutoff,k, AR):
    rhs = np.zeros_like(particles)
    n_part=len(particles)
    acc = point_in_active_region(particles, AR)
    Dij = squareform(pdist(particles)) * np.outer(acc,acc)
    Dij = (Dij>lower_cutoff) * (Dij<cutoff)
    for i in range(n_part):
        for j in range(i+1,n_part):
            if Dij[i,j]!=0:
                rhs[i] += -k*(particles[i]-particles[j])
                rhs[j] += +k*(particles[i]-particles[j])
    return rhs

@numba.jit
def integrate_one_timestep(particles, velocities, dt, m , cutoff, lower_cutoff, k, AR, drag_factor):
    particles = particles + dt * velocities
    velocities = (1-drag_factor)*velocities + dt/m * RHS(particles,cutoff=cutoff, lower_cutoff=lower_cutoff,k=k, AR=AR)
    return particles, velocities
