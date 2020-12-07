import numba
import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .pair_interactions import RHS
from scipy.spatial.distance import pdist, squareform
import sys

def get_linear_grid(sim_info,res=32):
    return np.linspace(sim_info['x_min'],sim_info['x_max'],res), np.linspace(sim_info['y_min'],sim_info['y_max'],res)

def get_grid_pairs(sim_info,res=32):
    gridX, gridY = get_linear_grid(sim_info,res)
    XY = np.array(np.meshgrid(gridX,gridY)).reshape(2,res*res)
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
    return fVs/(4*np.pi*mu)

def fVs_on_grid(pXs, pVs, sim_info, mu=1, res=32):
    fXs = get_grid_pairs(sim_info, res).T
    return fXs, fVs_on_points(fXs, pXs, pVs, mu=mu)

def fVs_on_particles(pXs, pVs, sim_info, mu=1, res=32, spline_degree=3):
    fXs_grid, fVs_grid = fVs_on_grid(pXs, pVs, sim_info, mu, res=res)
    gridX, gridY = get_linear_grid(sim_info,res)
    func_fV_x = RectBivariateSpline(gridX, gridY, fVs_grid[:,0].reshape(res,res).T, kx=spline_degree, ky=spline_degree) # Currently using a spline_degreerd degree splin
    func_fV_y = RectBivariateSpline(gridX, gridY, fVs_grid[:,1].reshape(res,res).T, kx=spline_degree, ky=spline_degree)
    fVs_x = func_fV_x.ev(pXs[:,0], pXs[:,1])
    fVs_y = func_fV_y.ev(pXs[:,0], pXs[:,1])
    return np.array((fVs_x,fVs_y)).T

#@numba.jit ##TODO reimplement this to work with numba
def particle_fusion(pXs, pVs, ms, acc, n_part, n_fuse, minit):
    """
    This function handles the fusion of particles due to an agglomeration of mass in an aster
    """
    ind_f = []
    ## Get the distance between the activated particles (not in square form, because we need the list of non-zero entries, only occuring once)
    dist_among_acc = pdist(pXs[np.nonzero(acc)])

    ## This logic block should never be false in a real-case scenario. It occurs when more particles should be fused together to ensure constant density than there are activated particles
    SKIP_TO_NEXT_STEP = False
    if n_fuse < sum(acc):
        ## Determine the n_fuse minimal distances among the activated particles. These distances are between the particles we want to fuse
        try:
            n_fuse_minimal_vals = np.partition(dist_among_acc,n_fuse)[:n_fuse]
        except:
            print(f'Something went wrong with trying to partition... n_fuse = {n_fuse}, number of activated particles: {sum(acc)}')
            n_fuse_minimal_vals = []
            SKIP_TO_NEXT_STEP = True
    else:
        print('Warning: Did enough activated particles to fuse. Reducing number of fused particles')
        n_fuse = int(sum(acc))
        n_fuse_minimal_vals = dist_among_acc
        SKIP_TO_NEXT_STEP = True

    ## Now go over all particles (because we need the proper indices)
    Dij = squareform(pdist(pXs)) ## Calculate some values twice, but should not be a large problem
    cnt = 0 # Accounting how many fusion processes we did
    for i in range(n_part):
        if acc[i]:
            for j in range(i+1,n_part):
                ## Check if we found a pair that we want to fuse
                if Dij[i,j] in n_fuse_minimal_vals and acc[j]:
                    ## Inelastic collision math
                    pXs[i,:] = (pXs[i,:]+pXs[i,:])/2
                    pVs[i,:] = (ms[i]*pXs[i,:]+ms[j]*pXs[j,:])/(ms[i]+ms[j])
                    ms[i] = ms[i]+ms[j]
                    ind_f.append(j) ## particle j can be respawned
                    ms[j]=minit
                    acc[j] = 0 ## make sure to not double do j in a new iteration
                    cnt += 1
                    break # Make sure not to fuse particle j any more
    if cnt == n_fuse or SKIP_TO_NEXT_STEP: ## This should be the regular exit of this function
        return pXs, pVs, ms, acc, ind_f
    elif cnt < n_fuse: ## Some particles merge more than once. This catches this behavior
        pXs, pVs, ms, acc, ind_f_tmp = particle_fusion(pXs, pVs, ms, acc, n_part, n_fuse-cnt, minit)
        ind_f = ind_f+ind_f_tmp
        return pXs, pVs, ms, acc, ind_f
    else: ## No idea what happens here. Should never happen. Raise error when this is encountered
        raise RuntimeError(f'Something went wrong in particle_fusion. Merged more particles ({cnt}) than required ({n_fuse})...')

def create_and_destroy_particles(pXs, pVs, acc, ms, _config, sim_info):
    """
    This function handles the destruction and spawning of new particles in the simulation when the field of view moves. This is achieved by several sub logics. One is that particles which leave the field of view can be ignored and set to a new particle positon. Furthermore, particles that agglomerate in the center due to the aster formation can be fused together. This is done adaptively, so that the fusion is only performed on so many particles to have the density of particles fixed in the newly spawned area.
    """
    ## TODO generalize for any velocity vector
    ## Get the required parameters to determine the new geometry
    dt = _config['dt']
    L = _config['L']
    vx = _config['window_velocity'][0]
    vy = _config['window_velocity'][1]
    assert(vx >= 0) # at least for now
    assert(vy >= 0)

    ## Determine the new geometry as well as the old one
    x_min_new = sim_info['x_min']
    x_min_old = x_min_new-dt*vx
    x_max_new = sim_info['x_max']
    x_max_old = x_max_new-dt*vx
    y_min_new = sim_info['y_min']
    y_min_old = y_min_new-dt*vy
    y_max_new = sim_info['y_max']
    y_max_old = y_max_new-dt*vy

    ## Determine how many particles have to be spawned based on the density
    new_area =   (x_max_new - x_min_new) * (y_max_new - y_max_old) \
               + (x_max_new - x_max_old) * (y_max_new - y_min_new) \
               - (x_max_new - x_max_old) * (y_max_new - y_max_old)
    n_particles_to_spawn = _config['particle_density'] * new_area

    ## Which particles left the field of view and can now be reset
    ind_x = np.nonzero( pXs[:,0]<x_min_new )[0]
    ind_y = np.nonzero( pXs[:,1]<y_min_new )[0]

    ## Determine how many particles need to be fused in the activated area to keep the density constant
    ind_f = []
    n_particles_to_spawn -= len(ind_x) + len(ind_y)
    tmp = int(n_particles_to_spawn)
    n_particles_to_spawn = tmp + int(np.random.rand()<(n_particles_to_spawn-tmp))

    ## Fusion process (delegated into it's own function, because reasonably difficult logic there)
    if _config['const_particle_density'] and n_particles_to_spawn > 0 and sim_info['time_step_index']>0:
        pXs, pVs, ms, acc, ind_f = particle_fusion(pXs, pVs, ms, acc, n_part=_config['n_part'], n_fuse = n_particles_to_spawn, minit=_config['m_init'])

    ## Set new positions in the newly spawned areas for reset particles
    pXs[ind_x,0] = np.random.rand(len(ind_x))*(x_max_new-x_max_old)+x_max_old
    pXs[ind_x,1] = np.random.rand(len(ind_x))*(y_max_new-y_min_new)+y_min_new
    pXs[ind_y,0] = np.random.rand(len(ind_y))*(x_max_new-x_min_new)+x_min_new
    pXs[ind_y,1] = np.random.rand(len(ind_y))*(y_max_new-y_max_old)+y_max_old
    if vx > 0:
        pXs[ind_f,0] = np.random.rand(len(ind_f))*(x_max_new-x_max_old)+x_max_old
        pXs[ind_f,1] = np.random.rand(len(ind_f))*(y_max_new-y_min_new)+y_min_new
    elif vy > 0:
        pXs[ind_f,0] = np.random.rand(len(ind_f))*(x_max_new-x_min_new)+x_min_new
        pXs[ind_f,1] = np.random.rand(len(ind_f))*(y_max_new-y_max_old)+y_max_old

    ## New particles have zero velocity...
    pVs[ind_x] = np.zeros(shape=(len(ind_x),2))
    pVs[ind_y] = np.zeros(shape=(len(ind_y),2))
    pVs[ind_f] = np.zeros(shape=(len(ind_f),2))
    ## ... and no activation
    acc[ind_x] = np.zeros(shape=len(ind_x))
    acc[ind_y] = np.zeros(shape=len(ind_y))
    acc[ind_f] = np.zeros(shape=len(ind_f))
    return pXs, pVs, acc

def periodic_boundary(pXs, pVs, acc, _config, sim_info):
    """
    This function handles the periodic boundary functions on a non-moving image
    """
    x_min = sim_info['x_min']
    x_max = sim_info['x_max']
    y_min = sim_info['y_min']
    y_max = sim_info['y_max']
    x_length = x_max - x_min
    y_length = y_max - y_min

    ind_x = np.nonzero((pXs[:, 0] < x_min) | (pXs[:, 0] > x_max))[0]
    ind_y = np.nonzero((pXs[:, 1] < y_min) | (pXs[:, 1] > y_max))[0]

    pXs[ind_x, 0] = (pXs[ind_x, 0] - x_min) % x_length + x_min
    pXs[ind_y, 1] = (pXs[ind_y, 1] - y_min) % y_length + y_min

    return pXs, pVs, acc

def integrate_one_timestep(pXs,
                           pVs,
                           acc,
                           ms,
                           sim_info,
                           _config,
                           activation_fn,
                           get_fluid_velocity=False,
                           use_interpolated_fluid_velocities=True,
                           DEBUG_INTERPOLATION=False):
    dt = _config['dt']
    Rdrag = _config['Rdrag']
    mu = _config['mu']
    pXs = pXs + dt * pVs
    rhs, acc = RHS(pXs, acc,activation_fn, _config=_config)
    pVs = (1-_config['drag_factor'])*pVs + dt * rhs / ms[:,np.newaxis]
    if _config.get('periodic_boundary', False):
        pXs, pVs, acc = periodic_boundary(pXs, pVs, acc, _config, sim_info)
    if _config['brownian_motion_delta'] > 0:
         pVs += _config['brownian_motion_delta'] * np.sqrt(_config['dt'])*np.random.normal(size=pXs.shape) / _config['dt'] # so that the average dx scales with sqrt(dt)
    if 'window_velocity' in _config and np.linalg.norm(_config['window_velocity']) > 0:
        pXs, pVs, acc = create_and_destroy_particles(pXs, pVs, acc, ms, _config, sim_info)
    if Rdrag > 0:
        raise RuntimeError('Fluids implementation is broken, please do not use this module.')
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
        pVs += (6*np.pi*mu*Rdrag*fVs)*dt/ms[:,np.newaxis]
    if get_fluid_velocity:
        fXs, fVs = fVs_on_grid(pXs, pVs, sim_info=sim_info, mu=mu)
        return pXs, pVs, acc, ms, fXs, fVs
    else:
        return pXs, pVs, acc, ms, None, None
