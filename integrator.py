import numba
import numpy as np
from scipy.spatial.distance import pdist, squareform

@numba.njit(parallel=True)
def point_in_active_region(ps, AR):
    return (ps[:,0]>-1) * (ps[:,0]<1) * (ps[:,1]>-AR) * (ps[:,1]<AR)

@numba.jit
def RHS(pXs, cutoff, lower_cutoff,k, AR, r0):
    rhs = np.zeros_like(pXs)
    n_part=len(pXs)
    acc = point_in_active_region(pXs, AR)
    Dij = squareform(pdist(pXs))
    Iij = Dij * np.outer(acc,acc)
    Iij = (Iij>lower_cutoff) * (Iij<cutoff)
    for i in range(n_part):
        for j in range(i+1,n_part):
            if Iij[i,j]!=0:
                rhs[i] += -k*((Dij[i,j]-r0)/Dij[i,j])*(pXs[i]-pXs[j])
                rhs[j] += +k*((Dij[i,j]-r0)/Dij[i,j])*(pXs[i]-pXs[j])
    return rhs

@numba.jit
def get_grid_pairs(L,res=32):
    gridX = np.linspace(-L,L,32)
    XY = np.array(np.meshgrid(gridX,gridX)).reshape(2,res*res).T
    return XY

@numba.jit
def fVs_on_points(ps, pXs, pVs, mu=1):
    fVs = np.zeros_like(ps)
    for p,v in zip(pXs,pVs):
        dX = ps-p
        l=np.linalg.norm(dX, axis=1)
        ind = np.nonzero(l) # Only update when the norm is non vanishing (important when ps == pXs)
        fVs[ind] += (np.outer(-np.log(l),v) + np.multiply(dX.T,np.dot(dX,v)/l**2).T)[ind]
    return fVs/(8*np.pi*mu)

@numba.jit
def fVs_on_grid(pXs, pVs, L, mu=1):
    fXs = get_grid_pairs(L)
    return fXs, fVs_on_points(fXs, pXs, pVs, mu=mu)

@numba.jit
def integrate_one_timestep(pXs, pVs, dt, m , cutoff, lower_cutoff, k, AR, drag_factor, r0, L, get_fluid_velocity=False):
    pXs = pXs + dt * pVs
    pVs = (1-drag_factor)*pVs + dt/m * RHS(pXs,cutoff=cutoff, lower_cutoff=lower_cutoff,k=k, AR=AR, r0=r0)
    if get_fluid_velocity:
        fXs, fVs = fVs_on_grid(pXs, pVs, L)
        return pXs, pVs, fXs, fVs
    else:
        return pXs, pVs, None, None
