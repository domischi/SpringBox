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

MAKE_VIDEO=True
SAVEFIG=True

ex = Experiment('SpringBox')
if SAVEFIG or MAKE_VIDEO:
    ex.observers.append(FileStorageObserver.create('data'))
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def cfg():
    AR=3/4
    L=1.5
    n_part=5000
    k=1
    ## Matt's values
    cutoff = 2.5/4
    lower_cutoff = 0.1/4
    dt=.005
    m=1.
    T=4
    savefreq = 10
    r0=0.2
    drag_factor=1

@ex.automain
def main(AR, n_part, cutoff, dt, m,T,k, savefreq, L, drag_factor,lower_cutoff, r0):
    imagefolder = f'/tmp/boxspring-{int(time.time())}'
    os.makedirs(imagefolder)
    pXs = (np.random.rand(n_part,2)-.5)*2*L
    pVs = np.zeros_like(pXs)
    for i in tqdm(range(int(T/dt))):
        pXs, pVs, fXs, fVs = integrate_one_timestep(pXs, pVs, dt=dt, m=m,cutoff=cutoff,lower_cutoff=lower_cutoff,k=k,AR=AR, drag_factor=drag_factor, r0=r0, L=L)
        if savefreq!=None and i%savefreq == 0:
            plot_points(pXs, pVs, i, cutoff=cutoff,lower_cutoff=lower_cutoff, image_folder=imagefolder, title=f'cutoff={cutoff:.2f}, t={i*dt:.3f}', AR=AR, L=L, SAVEFIG=SAVEFIG, ex=ex)
            plot_fluid(fXs, fVs, i, imagefolder, title=f'cutoff={cutoff:.2f}, t={i*dt:.3f}',L=L, SAVEFIG=False)
    if MAKE_VIDEO:
        video_path = generate_video_from_png(imagefolder)
        ex.add_artifact(video_path)
