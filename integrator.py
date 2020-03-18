import numba
import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from pair_interactions import RHS

def get_linear_grid(sim_info,res=32):
    return np.linspace(sim_info['x_min'],sim_info['x_max'],res), np.linspace(sim_info['y_min'],sim_info['y_max'],res)

def get_grid_pairs(sim_info,res=32):
    gridX, gridY = get_linear_grid(sim_info,res)
    XY = np.array(np.meshgrid(gridX,gridY)).reshape(2,res*res).T
    return XY

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

def fVs_on_grid(pXs, pVs, sim_info, mu=1, res=32):
    fXs = get_grid_pairs(sim_info, res)
    return fXs, fVs_on_points(fXs, pXs, pVs, mu=mu)

def fVs_on_particles(pXs, pVs, sim_info, mu=1, res=32, spline_degree=3):
    fXs_grid, fVs_grid = fVs_on_grid(pXs, pVs, sim_info, mu, res=res)
    gridX, gridY = get_linear_grid(sim_info,res)
    func_fV_x = RectBivariateSpline(gridX, gridY, fVs_grid[:,0].reshape(res,res).T, kx=spline_degree, ky=spline_degree) # Currently using a spline_degreerd degree splin
    func_fV_y = RectBivariateSpline(gridX, gridY, fVs_grid[:,1].reshape(res,res).T, kx=spline_degree, ky=spline_degree)
    fVs_x = func_fV_x.ev(pXs[:,0], pXs[:,1])
    fVs_y = func_fV_y.ev(pXs[:,0], pXs[:,1])
    return np.array((fVs_x,fVs_y)).T

def create_and_destroy_particles(pXs, pVs, _config, sim_info):
    ## TODO generalize for any velocity vector
    dt = _config['dt']
    L = _config['L']
    vx = _config['window_velocity'][0]
    vy = _config['window_velocity'][1]
    assert(vx > 0) # at least for now
    assert(vy == 0)
    x_min_new = sim_info['x_min']
    x_min_old = x_min_new-dt*vx
    x_max_new = sim_info['x_max']
    x_max_old = x_min_new-dt*vx
    ind_x = np.nonzero( pXs[:,0]<x_min_new )
    pXs[ind_x,0] = np.random.rand(len(ind_x))*(x_max_new-x_max_old)+x_max_old
    pXs[ind_x,1] = np.random.rand(len(ind_x))*2*L-L
    pVs[ind_x] = np.zeros(shape=(len(ind_x),2))
    return pXs, pVs

def integrate_one_timestep(pXs, pVs, acc, activation_fn, sim_info, _config, get_fluid_velocity=False, use_interpolated_fluid_velocities=True, DEBUG_INTERPOLATION=False):
    dt = _config['dt']
    Rdrag = _config['Rdrag']
    mu = _config['mu']
    pXs = pXs + dt * pVs
    rhs, acc = RHS(pXs, acc,activation_fn, _config=_config)
    pVs = (1-_config['drag_factor'])*pVs + dt/_config['m'] * rhs
    if _config['brownian_motion_delta'] > 0:
         pVs += _config['brownian_motion_delta'] * np.sqrt(_config['dt'])*np.random.normal(size=pXs.shape) / _config['dt'] # so that the average dx scales with sqrt(dt)
    if np.linalg.norm(_config['window_velocity']) > 0:
        pXs, pVs = create_and_destroy_particles(pXs, pVs, _config, sim_info)
    if Rdrag > 0:
        if use_interpolated_fluid_velocities:
            fVs = fVs_on_particles(pXs, pVs, sim_info=sim_info, res=32, spline_degree=3, mu=mu)
        else:
            fVs = fVs_on_points(pXs, pXs, pVs, mu=mu)
        if DEBUG_INTERPOLATION:
            if use_interpolated_velocities:
                fVs2 = fVs_on_points(pXs, pXs, pVs, mu=mu)
            else:
                fVs2 = fVs_on_particles(pXs, pVs, sim_info=sim_info, res=32, spline_degree=3, mu=mu)
            plt.quiver(pXs[:,0],pXs[:,1], fVs [:,0], fVs [:,1], color='red')
            plt.quiver(pXs[:,0],pXs[:,1], fVs2[:,0], fVs2[:,1], color='green')
            plt.show(block=True)
        pVs += 6*np.pi*mu*Rdrag*fVs
    if get_fluid_velocity:
        fXs, fVs = fVs_on_grid(pXs, pVs, sim_info=sim_info, mu=mu)
        return pXs, pVs, acc, fXs, fVs
    else:
        return pXs, pVs, acc, None, None
