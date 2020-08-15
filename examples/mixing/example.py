from sacred import Experiment, SETTINGS
from sacred.dependencies import PackageDependency
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from functools import partial
from copy import deepcopy
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, ncols=100)
import numpy as np
import time
import datetime
import os
import numba
from numba.errors import NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)
import SpringBox
from SpringBox.integrator import integrate_one_timestep
from SpringBox.activation import *
from SpringBox.post_run_hooks import post_run_hooks
from SpringBox.measurements import do_measurements, do_one_timestep_correlation_measurement, get_mixing_score

ex = Experiment('SpringBox')
#ex.observers.append(MongoObserver.create())
ex.observers.append(FileStorageObserver.create(f'data/'))
ex.dependencies.add(PackageDependency("SpringBox",SpringBox.__version__))

@ex.config
def cfg():
    ## Simulation parameters
    sweep_experiment = False
    mixing_experiment = True
    run_id = 0
    savefreq_fig = 3
    savefreq_data_dump = 3
    # Speeds up the computation somewhat, but incurs an error due to oversmoothing of fluids (which could however be somewhat physical)
    use_interpolated_fluid_velocities = True
    dt=.01
    T=1
    particle_density = 31.25
    MAKE_VIDEO = True
    SAVEFIG    = False
    const_particle_density = False
    measure_one_timestep_correlator = False
    periodic_boundary = True

    ## Geometry parameters / Activation Fn
    activation_fn_type = 'const-rectangle' # For the possible choices, see the activation.py file
    AR=.75
    L=2
    n_part = int(particle_density * ((2*L)**2))
    if mixing_experiment:
        assert (n_part % 2 == 0)

    ## Interaction parameters
    # Particle properties
    m_init=1.
    activation_decay_rate = 10. # Ex. at dt=0.01 this leads to an average deactivation of 10% of the particles
    # Spring properties
    spring_cutoff = 50./np.sqrt(n_part) # Always have a same average of particles that interact
    spring_lower_cutoff = spring_cutoff/25
    spring_k=1.
    spring_r0=0.2
    # LJ properties
    LJ_eps=0.
    LJ_r0=.05
    LJ_cutoff=2.5/1.122*LJ_r0 # canonical choice
    # Brownian properties
    brownian_motion_delta = 0.

    ## Fluid parameters
    mu=10.
    Rdrag = .01
    drag_factor=1

def get_sim_info(old_sim_info, _config, i):
    sim_info = old_sim_info
    dt = _config['dt']
    L = _config['L']
    T = _config['T']
    savefreq_fig = _config['savefreq_fig']
    savefreq_dd = _config['savefreq_data_dump']
    sim_info['t']     = i*dt
    sim_info['time_step_index'] = i
    sim_info['x_min'] = -L
    sim_info['y_min'] = -L
    sim_info['x_max'] =  L
    sim_info['y_max'] =  L
    sim_info['plotting_this_iteration'] = (savefreq_fig!=None and i%savefreq_fig == 0)
    sim_info['data_dump_this_iteration'] = (savefreq_dd!=None and (i%savefreq_dd == 0 or i==int(T/dt)-1))
    sim_info['get_fluid_velocity_this_iteration'] = sim_info['plotting_this_iteration'] or sim_info['data_dump_this_iteration']
    sim_info['measure_one_timestep_correlator'] = ( 'measure_one_timestep_correlator' in _config.keys() and _config['measure_one_timestep_correlator'])
    return sim_info

@ex.automain
def main(_config, _run):
    ## Load local copies of the parameters needed in main
    run_id = _config['run_id']

    ## Setup Folders
    timestamp = int(time.time())
    data_dir = f'/tmp/boxspring-{run_id}-{timestamp}'
    os.makedirs(data_dir)

    ## Initialize particles
    pXs = (np.random.rand(_config['n_part'],2)-.5)*2*_config['L']
    pXs[:_config['n_part']//2,0] = -np.random.rand(_config['n_part']//2)*_config['L']
    pXs[_config['n_part']//2:,0] = +np.random.rand(_config['n_part']//2)*_config['L']
    pVs = np.zeros_like(pXs)
    acc = np.zeros(len(pXs))
    ms  = _config['m_init']*np.ones(len(pXs))

    if _config['use_interpolated_fluid_velocities']:
        print('WARNING: Using interpolated fluid velocities can yield disagreements. The interpolation is correct for most points. However, for some the difference can be relatively large.')

    ## Initialize information dict
    sim_info = {'data_dir': data_dir}

    ## Integration loop
    N_steps = int(_config['T']/_config['dt'])
    for i in tqdm(range(N_steps), disable = True):
        if _config['sweep_experiment'] and (i%50)==0:
            print(f"[{datetime.datetime.now()}] Run {_config['run_id']}: Doing step {i+1: >6} of {N_steps}")

        sim_info = get_sim_info(sim_info, _config, i)
        activation_fn = activation_fn_dispatcher(_config, sim_info['t'])
        if sim_info['measure_one_timestep_correlator']:
            pXs_old = deepcopy(pXs)
        pXs, pVs, acc, ms, fXs, fVs = integrate_one_timestep(pXs = pXs,
                                                             pVs = pVs,
                                                             acc = acc,
                                                             ms  = ms,
                                                             activation_fn = activation_fn,
                                                             sim_info = sim_info,
                                                             _config = _config,
                                                             get_fluid_velocity=sim_info['get_fluid_velocity_this_iteration'],
                                                             use_interpolated_fluid_velocities=_config['use_interpolated_fluid_velocities'])

        do_measurements(ex = ex,
                        _config = _config,
                        _run = _run,
                        sim_info = sim_info,
                        pXs = pXs,
                        pVs = pVs,
                        acc = acc,
                        ms = ms,
                        fXs = fXs,
                        fVs = fVs,
                        plotting_this_iteration = sim_info['plotting_this_iteration'],
                        save_all_data_this_iteration = sim_info['data_dump_this_iteration'])
        if sim_info['measure_one_timestep_correlator']:
            do_one_timestep_correlation_measurement(ex = ex,
                                                    _config = _config,
                                                    _run = _run,
                                                    sim_info = sim_info,
                                                    pXs = pXs,
                                                    pXs_old = pXs_old)
        print(get_mixing_score(pXs, _config, sim_info))
    post_run_hooks(ex, _config, _run, data_dir)
