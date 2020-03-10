from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from tqdm import tqdm
import numpy as np
import sys
import time
import os
import numba
from illustration import *
from integrator import *

MAKE_VIDEO= True
SAVEFIG   = False

ex = Experiment('SpringBox')
if SAVEFIG or MAKE_VIDEO:
    ex.observers.append(FileStorageObserver.create('data'))
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def cfg():
    ARs=[ 1. ]
    L=2
    n_part=5000
    cutoff = 50./np.sqrt(n_part) # Always have a same average of particles that interact
    lower_cutoff = cutoff/25
    k=1.
    dt=.005
    m=1.
    mu=1.
    Rdrag = .01
    T=4
    savefreq = 10
    r0=0.2
    drag_factor=1

@ex.automain
def main(ARs, n_part, cutoff, dt, m,T,k, savefreq, L, drag_factor,lower_cutoff, r0, mu, Rdrag):
    for AR in tqdm(ARs):
        timestamp = int(time.time())
        image_folder = f'/tmp/boxspring-{timestamp}'
        os.makedirs(image_folder)
        pXs = (np.random.rand(n_part,2)-.5)*2*L
        pVs = np.zeros_like(pXs)
        for i in tqdm(range(int(T/dt))):
            plotting_this_iteration = savefreq!=None and i%savefreq == 0
            pXs, pVs, fXs, fVs = integrate_one_timestep(pXs, pVs, dt=dt, m=m,cutoff=cutoff,lower_cutoff=lower_cutoff,k=k,AR=AR, drag_factor=drag_factor, r0=r0, L=L, mu=mu, Rdrag=Rdrag, get_fluid_velocity=plotting_this_iteration)
            if plotting_this_iteration:
                plot_data(pXs, pVs, fXs, fVs, i, image_folder=image_folder, title=f't={i*dt:.3f}', L=L, fix_frame=True, SAVEFIG=SAVEFIG, ex=ex, plot_particles=True, plot_fluids=True, side_by_side=True, fluid_plot_type = 'quiver')
        if MAKE_VIDEO:
            video_path = generate_video_from_png(image_folder)
            ex.add_artifact(video_path, name=f'video-{AR:.2f}.avi')
