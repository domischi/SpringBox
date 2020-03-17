import numba
import numpy as np
from scipy.spatial.distance import pdist, squareform

@numba.njit(parallel=True)
def point_in_active_region(ps, AR):
    return (ps[:,0]>-1) * (ps[:,0]<1) * (ps[:,1]>-AR) * (ps[:,1]<AR)

@numba.jit
def active_particles(pXs, prv_acc, _config):
    # Particles that were previously active, now can loose their activation with a rate of activation_decay_rate
    prv_acc = prv_acc * (np.random.rand(len(prv_acc)) > _config['activation_decay_rate'] * _config['dt'])
    # Or particles can become active when they enter the light activated area
    new_acc = point_in_active_region(pXs, _config['AR'])
    return np.maximum(prv_acc, new_acc)

@numba.jit
def RHS(pXs, prv_acc, _config):
    rhs = np.zeros_like(pXs)
    n_part=_config['n_part']
    k=_config['spring_k']
    r0=_config['spring_r0']
    acc = active_particles(pXs, prv_acc, _config)
    Dij = squareform(pdist(pXs))
    Iij = Dij * np.outer(acc,acc)
    Iij = (Iij>_config['spring_lower_cutoff']) * (Iij<_config['spring_cutoff'])
    for i in range(n_part):
        for j in range(i+1,n_part):
            if Iij[i,j]!=0:
                rhs[i] += -k*((Dij[i,j]-r0)/Dij[i,j])*(pXs[i]-pXs[j])
                rhs[j] += +k*((Dij[i,j]-r0)/Dij[i,j])*(pXs[i]-pXs[j])
    return rhs, acc

