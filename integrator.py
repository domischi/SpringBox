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
        lp =  l[ind]
        dXp= dX[ind]
        fVs[ind] += np.outer(-np.log(lp),v) + np.multiply(dXp.T,np.dot(dXp,v)/lp**2).T
    return fVs/(8*np.pi*mu)

@numba.jit
def fVs_on_grid(pXs, pVs, L, mu=1):
    fXs = get_grid_pairs(L)
    return fXs, fVs_on_points(fXs, pXs, pVs, mu=mu)

@numba.jit
def integrate_one_timestep(pXs, pVs, dt, m , cutoff, lower_cutoff, k, AR, drag_factor, r0, L, mu, Rdrag, get_fluid_velocity=False):
    pXs = pXs + dt * pVs
    pVs = (1-drag_factor)*pVs + dt/m * RHS(pXs,cutoff=cutoff, lower_cutoff=lower_cutoff,k=k, AR=AR, r0=r0)
    if Rdrag > 0:
        # Rdrag determines the size of the fluid to object coupling. Each of the particles is assumed to follow 3D Stokes drag (F_d=6*pi*mu*Rdrag*v) where Rdrag is the radius of an equivalent sphere
    # This might be a bad approximation due to Stokes paradox, stating that the problem is ill defined in 2D
    # The drag in the 2D case is rather radius independent and depends on the Reynolds number. However, since all particles are assumed to be the same, and the Reynolds number stays constant as well, this parametrization is still valid.
        pVs += 6*np.pi*mu*Rdrag*fVs_on_points(pXs, pXs, pVs, mu=mu) # TODO implement this as a interpolation task rather than computing it on each and every particle separately.
    if get_fluid_velocity:
        fXs, fVs = fVs_on_grid(pXs, pVs, L, mu=mu)
        return pXs, pVs, fXs, fVs
    else:
        return pXs, pVs, None, None
