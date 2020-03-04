from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import sys
import time
import os
import numba
from videomaker import generate_video_from_png

MAKE_VIDEO=True
SAVEFIG=False

ex = Experiment('SpringBox')
if SAVEFIG:
    ex.observers.append(FileStorageObserver.create('data'))
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def cfg():
    AR=1
    L=1
    n_part=500
    k=1
    epsilon=1
    cutoff = L/np.sqrt(n_part)*2*(1+epsilon)
    lower_cutoff = cutoff/25 # Thats what Matt used in his matlab code, I am not sure why though...
    dt=.01
    m=1.
    T=10.
    savefreq = None
    drag_factor=1

#@ex.capture
@numba.njit(parallel=True)
def point_in_active_region(p, AR):
    return int(p[0]>-1 and p[0]<1 and p[1]>-AR and p[1]<AR)

@ex.capture
def plot_points(particles, velocities, i,cutoff,lower_cutoff, image_folder, t, AR,L, show_springs=False, fix_frame=True):
    fig=plt.figure(figsize=(12,10))
    vabs = np.linalg.norm(velocities, axis=1)
    sc=plt.scatter(particles[:,0],particles[:,1], c=vabs, cmap=plt.get_cmap('hot'), vmin=0, vmax=2*max(vabs))
    plt.colorbar(sc)
    if show_springs:
        for p1 in tqdm(particles, total=len(particles), leave=False):
            for p2 in particles:
                if (    np.linalg.norm(p1-p2)<cutoff 
                    and np.linalg.norm(p1-p2)>lower_cutoff 
                    and point_in_active_region(p1, AR) 
                    and point_in_active_region(p2, AR)):
                    plt.plot( (p1[0],p2[0]) , (p1[1],p2[1]), color='k')
    #plt.show()
    plt.title(f't={t:.2f}')
    if fix_frame:
        plt.xlim([-L,L])
        plt.ylim([-L,L])
    IMG_NAME=f'{image_folder}/fig{i:08}.png'
    plt.savefig(IMG_NAME)
    if SAVEFIG:
        ex.add_artifact(IMG_NAME)
    try:
        plt.close(fig)
    except:
        print('Something went wrong with closing the figure')
        pass

@numba.njit
def RHS(particles, cutoff, lower_cutoff,k, AR):
    rhs = np.zeros_like(particles)
    n_part=len(particles)
    for i in range(n_part):
        p1 = particles[i]
        p1_acc = point_in_active_region(p1, AR) 
        for j in range(i+1,n_part):
            p2 = particles[j]
            p2_acc = point_in_active_region(p2, AR)
            d = np.linalg.norm(p1-p2)
            if (    d<cutoff 
                and d>lower_cutoff):
                rhs[i] += -k*p1_acc*p2_acc*(p1-p2)
                rhs[j] += +k*p1_acc*p2_acc*(p1-p2)
    return rhs

@ex.automain
def main(AR, n_part, cutoff, dt, m,T,k, savefreq, L, drag_factor,lower_cutoff):
    imagefolder = f'/tmp/boxspring-{int(time.time())}'
    os.makedirs(imagefolder)
    particles = (np.random.rand(n_part,2)-.5)*2*L
    velocities = np.zeros_like(particles)
    for i in tqdm(range(int(T/dt))):
        particles = particles + dt * velocities
        velocities = (1-drag_factor)*velocities + dt/m * RHS(particles,cutoff=cutoff, lower_cutoff=lower_cutoff,k=k, AR=AR) ## this should be the infinite drag case, right
        if savefreq!=None and i%savefreq == 0:
            plot_points(particles, velocities, i, cutoff,lower_cutoff, imagefolder, t=i*dt)
    if MAKE_VIDEO:
        video_path = generate_video_from_png(imagefolder)
        if SAVEFIG:
            ex.add_artifact(video_path)
