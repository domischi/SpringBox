from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, ncols=100)
import numpy as np
import sys
import time
import os
import numba
from illustration import *
from integrator import *
import multiprocessing

MAKE_VIDEO= True
SAVEFIG   = False

ex = Experiment('SpringBox')
if SAVEFIG or MAKE_VIDEO:
    ex.observers.append(FileStorageObserver.create('data'))
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def cfg():
    run_id = 0 
    AR = 1.
    L=2
    n_part=5000
    cutoff = 50./np.sqrt(n_part) # Always have a same average of particles that interact
    lower_cutoff = cutoff/25
    k=1.
    dt=.02
    m=1.
    mu=1.
    Rdrag = .01
    T=4
    savefreq = 10
    r0=0.2
    drag_factor=1
    # Speeds up the computation somewhat, but incurs an error due to oversmoothing of fluids (which could however be somewhat physical)
    use_interpolated_fluid_velocities = True

#TODO go for less magic....
@ex.automain
def main(run_id, AR, n_part, cutoff, dt, m,T,k, savefreq, L, drag_factor,lower_cutoff, r0, mu, Rdrag, use_interpolated_fluid_velocities):
    timestamp = int(time.time())
    image_folder = f'/tmp/boxspring-{run_id}-{timestamp}'
    os.makedirs(image_folder)
    pXs = (np.random.rand(n_part,2)-.5)*2*L
    pVs = np.zeros_like(pXs)
    if use_interpolated_fluid_velocities:
        print('WARNING: Using interpolated fluid velocities can yield disagreements. The interpolation is correct for most points. However, for some the difference can be relatively large.')
    time.sleep(3) # To have the output of all other runs finished
    for i in tqdm(range(int(T/dt)), position=run_id):
        plotting_this_iteration = savefreq!=None and i%savefreq == 0
        pXs, pVs, fXs, fVs = integrate_one_timestep(pXs, pVs, dt=dt, m=m,cutoff=cutoff,lower_cutoff=lower_cutoff,k=k,AR=AR, drag_factor=drag_factor, r0=r0, L=L, mu=mu, Rdrag=Rdrag, get_fluid_velocity=plotting_this_iteration, use_interpolated_fluid_velocities=use_interpolated_fluid_velocities)
        if plotting_this_iteration:
            plot_data(pXs, pVs, fXs, fVs, i, image_folder=image_folder, title=f't={i*dt:.3f}', L=L, fix_frame=True, SAVEFIG=SAVEFIG, ex=ex, plot_particles=True, plot_fluids=True, side_by_side=True, fluid_plot_type = 'quiver')
    if MAKE_VIDEO:
        video_path = generate_video_from_png(image_folder)
        ex.add_artifact(video_path, name=f'video-{AR:.2f}.avi')
