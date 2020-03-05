from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy.spatial.distance import pdist, squareform
import sys
import time
import os
import numba
from videomaker import generate_video_from_png

MAKE_VIDEO=True
SAVEFIG=True

ex = Experiment('SpringBox')
if SAVEFIG:
    ex.observers.append(FileStorageObserver.create('data'))
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def cfg():
    AR=3/4
    L=1.5
    n_part=5000
    k=1
    epsilon=1
    #cutoff = L/np.sqrt(n_part)*2*(1+epsilon)
    #lower_cutoff = cutoff/25 # Thats what Matt used in his matlab code, I am not sure why though...
    cutoff = 2.5/4
    lower_cutoff = 0.1/4
    dt=.0005
    m=1.
    T=10.
    savefreq = 10
    drag_factor=1

@numba.njit(parallel=True)
def point_in_active_region(p, AR):
    return int(p[0]>-1 and p[0]<1 and p[1]>-AR and p[1]<AR)

@numba.njit(parallel=True)
def point_in_active_region_vec(ps, AR):
    return (ps[:,0]>-1) * (ps[:,0]<1) * (ps[:,1]>-AR) * (ps[:,1]<AR)

@ex.capture
def plot_points(particles, velocities, i,cutoff,lower_cutoff, image_folder, t, AR,L, show_springs=False, fix_frame=True):
    fig=plt.figure(figsize=(12,10))
    vabs = np.linalg.norm(velocities, axis=1)
    sc=plt.scatter(particles[:,0],particles[:,1], c=vabs, cmap=plt.get_cmap('viridis'), vmin=0, vmax=max(vabs))
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

@numba.jit
def RHS(particles, cutoff, lower_cutoff,k, AR):
    rhs = np.zeros_like(particles)
    n_part=len(particles)
    acc = point_in_active_region_vec(particles, AR)
    Dij = squareform(pdist(particles)) * np.outer(acc,acc)
    Dij = (Dij>lower_cutoff) * (Dij<cutoff)
    for i in range(n_part):
        for j in range(i+1,n_part):
            if Dij[i,j]!=0:
                rhs[i] += -k*(particles[i]-particles[j])
                rhs[j] += +k*(particles[i]-particles[j])
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
