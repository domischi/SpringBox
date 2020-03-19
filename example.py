from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, ncols=100)
import numpy as np
import sys
import time
import datetime
import os
import numba
from numba.errors import NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)
from illustration import *
from integrator import *
from activation import *
import multiprocessing

MAKE_VIDEO= True
SAVEFIG   = False

ex = Experiment('SpringBox')
if SAVEFIG or MAKE_VIDEO:
    ex.observers.append(MongoObserver.create())
    #ex.observers.append(FileStorageObserver.create(f'data/{str(datetime.date.today())}'))
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def cfg():
    ## Simulation parameters
    sweep_experiment = False
    run_id = 0
    savefreq = 3
    # Speeds up the computation somewhat, but incurs an error due to oversmoothing of fluids (which could however be somewhat physical)
    use_interpolated_fluid_velocities = True
    dt=.01
    T=4
    n_part=5000


    ## Geometry parameters / Activation Fn
    activation_fn_type = 'moving-circle' # For the possible choices, see the activation.py file
    activation_circle_radius = .5
    v_circ = np.array([.25, 0. ])
    x_0_circ = np.array([0,0])
    #AR = 1.
    L=2

    window_velocity = v_circ

    ## Interaction parameters
    # Particle properties
    m=1.
    activation_decay_rate = 10. # Ex. at dt=0.01 this leads to an average deactivation of 10% of the particles
    # Spring properties
    spring_cutoff = 50./np.sqrt(n_part) # Always have a same average of particles that interact
    spring_lower_cutoff = spring_cutoff/25
    spring_k=1.
    spring_r0=0.2
    # LJ properties
    LJ_eps=.1
    LJ_r0=.05
    LJ_cutoff=2.5/1.122*LJ_r0 # canonical choice
    # Brownian properties
    brownian_motion_delta = 0.01

    ## Fluid parameters
    mu=1.
    Rdrag = .05
    drag_factor=1


@ex.automain
def main(_config):
    ## Load local copies of the parameters needed in main
    dt = _config['dt']
    T = _config['T']
    n_part = _config['n_part']
    L = _config['L']
    savefreq = _config['savefreq']
    run_id = _config['run_id']
    vx = _config['window_velocity'][0]
    vy = _config['window_velocity'][1]

    ## Setup Folders
    timestamp = int(time.time())
    image_folder = f'/tmp/boxspring-{run_id}-{timestamp}'
    os.makedirs(image_folder)

    ## Initialize particles
    pXs = (np.random.rand(n_part,2)-.5)*2*L
    pVs = np.zeros_like(pXs)
    acc = np.zeros(len(pXs))

    if _config['use_interpolated_fluid_velocities']:
        print('WARNING: Using interpolated fluid velocities can yield disagreements. The interpolation is correct for most points. However, for some the difference can be relatively large.')

    time.sleep(3) # Required in multiprocessing environment to not overwrite any other output

    ## Initialize information dict
    sim_info = dict()

    ## Integration loop
    for i in tqdm(range(int(T/dt)), position=run_id, disable = _config['sweep_experiment']):
        plotting_this_iteration = savefreq!=None and i%savefreq == 0
        activation_fn = activation_fn_dispatcher(_config, i*dt)
        sim_info['t']     = i*dt
        sim_info['time_step_index'] = i
        sim_info['x_min'] = -L+dt*vx*i
        sim_info['y_min'] = -L+dt*vy*i
        sim_info['x_max'] =  L+dt*vx*i
        sim_info['y_max'] =  L+dt*vy*i
        pXs, pVs, acc, fXs, fVs = integrate_one_timestep(pXs = pXs,
                                                         pVs = pVs,
                                                         acc = acc,
                                                         activation_fn = activation_fn,
                                                         sim_info = sim_info,
                                                         _config = _config,
                                                         get_fluid_velocity=plotting_this_iteration,
                                                         use_interpolated_fluid_velocities=_config['use_interpolated_fluid_velocities'])
        if plotting_this_iteration:
            plot_data(pXs, pVs, fXs, fVs, sim_info, image_folder=image_folder, title=f't={i*dt:.3f}', L=_config['L'], fix_frame=True, SAVEFIG=SAVEFIG, ex=ex, plot_particles=True, plot_fluids=True, side_by_side=True, fluid_plot_type = 'quiver')
    if MAKE_VIDEO:
        video_path = generate_video_from_png(image_folder)
        ex.add_artifact(video_path, name=f"video.avi")
