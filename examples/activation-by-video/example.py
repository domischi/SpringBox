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
from SpringBox.measurements import do_measurements, do_one_timestep_correlation_measurement

ex = Experiment('SpringBox')
#ex.observers.append(MongoObserver.create())
ex.observers.append(FileStorageObserver.create(f'data/'))

activation_image = 'activation-*.png'
ex.dependencies.add(PackageDependency("SpringBox",SpringBox.__version__))

@ex.config
def cfg():
    ## Simulation parameters
    savefreq_fig = 1
    savefreq_data_dump = 1
    # Speeds up the computation somewhat, but incurs an error due to oversmoothing of fluids (which could however be somewhat physical)
    dt=.01
    T=8*dt
    particle_density = 310
    MAKE_VIDEO = True
    SAVEFIG    = True
    const_particle_density = False

    ## Geometry parameters / Activation Fn
    activation_fn_type = 'video' # For the possible choices, see the activation.py file
    activation_image_filepath = activation_image
    L=2
    n_part = particle_density * ((2*L)**2)

    ## Interaction parameters
    # Particle properties
    activation_decay_rate = np.Infinity
    # Spring properties
    spring_cutoff = 2.
    spring_lower_cutoff = 1e-3
    spring_k=1.

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
    sim_info['get_fluid_velocity_this_iteration'] = False
    sim_info['measure_one_timestep_correlator'] = _config.get('measure_one_timestep_correlator', False)
    return sim_info

@ex.automain
def main(_config, _run):
    ## Load local copies of the parameters needed in main
    run_id = _config.get('run_id', 0)

    ## Setup Folders
    timestamp = int(time.time())
    data_dir = f'/tmp/boxspring-{run_id}-{timestamp}'
    os.makedirs(data_dir)

    ## Initialize particles
    pXs = (np.random.rand(_config['n_part'],2)-.5)*2*_config['L']
    pVs = np.zeros_like(pXs)
    acc = np.zeros(len(pXs))
    ms  = _config.get('m_init', 1)*np.ones(len(pXs))

    if _config.get('use_interpolated_fluid_velocities', False):
        print('WARNING: Using interpolated fluid velocities can yield disagreements. The interpolation is correct for most points. However, for some the difference can be relatively large.')

    ## Initialize information dict
    sim_info = {'data_dir': data_dir}

    ## Integration loop
    N_steps = int(_config['T']/_config['dt'])
    for i in tqdm(range(N_steps), disable = _config.get('sweep_experiment', False)):
        if _config.get('sweep_experiment', False) and (i%50)==0:
            print(f"[{datetime.datetime.now()}] Run {_config['run_id']}: Doing step {i+1: >6} of {N_steps}")

        sim_info = get_sim_info(sim_info, _config, i)
        activation_fn = activation_fn_dispatcher(_config, sim_info['t'], experiment = ex)
        if sim_info.get('measure_one_timestep_correlator', False):
            pXs_old = deepcopy(pXs)
        pXs, pVs, acc, ms, fXs, fVs, M = integrate_one_timestep(pXs = pXs,
                                                             pVs = pVs,
                                                             acc = acc,
                                                             ms  = ms,
                                                             activation_fn = activation_fn,
                                                             sim_info = sim_info,
                                                             _config = _config,
                                                             get_fluid_velocity=sim_info['get_fluid_velocity_this_iteration'],
                                                             use_interpolated_fluid_velocities=_config.get('use_interpolated_fluid_velocities', False))

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
    post_run_hooks(ex, _config, _run, data_dir)
