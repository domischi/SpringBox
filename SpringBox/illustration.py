import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from .integrator import get_linear_grid

def get_fluid_colors(fXs, fVs, normalize=False):
    ## I realized that with limited data, this coloring does not make much sense
    """
    Given the velocity field (Ux, Uy) at positions (X,Y), compute for every x in (X,Y) if the position is contributing to the inflow or the outflow. This can be obtained by u*x (scalar product). If this quantity is positive, then it's an outflow, if it is negative it's an inflow.
    """
    ret = [np.dot(x,v) for x,v in zip(fXs, fVs)]
    if normalize:
        mi=min(ret)
        mx=max(ret)
        assert(mi<0 and mx>0)
        ret = [r/mi if r<0 else r/mx for r in ret]
    return np.array(ret)


def plot_fluid(ax, fXs, fVs, sim_info = dict() ,plot_type='streamplot', coloring_scheme='vabs'):
    plt.sca(ax)
    plt.title('Fluids')
    ng = int(np.sqrt(len(fXs)))
    if coloring_scheme == 'io':
        c = get_fluid_colors(fXs,fVs)
        cmap = 'bwr'
    elif coloring_scheme == 'solid':
        c = np.ones(len(fXs))
        cmap = 'binary'
    elif coloring_scheme == 'vabs':
        c = np.linalg.norm(fVs,axis=1)
        cmap = 'viridis'
    else:
        print('Illegal coloring_scheme in fluid plot. Check the code.')
        return
    if plot_type == 'quiver':
        p=plt.quiver(fXs[:,0],fXs[:,1], fVs[:,0],fVs[:,1], c, cmap=cmap , units='xy', pivot='mid')
    elif plot_type == 'streamplot':
        X,Y = get_linear_grid(sim_info)
        p=plt.streamplot(X,Y, fVs[:,0].reshape(ng,ng),fVs[:,1].reshape(ng,ng), color=c.reshape(ng,ng), cmap=cmap)
    else:
        print('Illegal type of fluid plot. Check the code.')
        return

def plot_points(ax, pXs, pVs):
    plt.sca(ax)
    plt.title('Particles')
    vabs = np.linalg.norm(pVs, axis=1)
    sc=plt.scatter(pXs[:,0],pXs[:,1], c=vabs, cmap=plt.get_cmap('viridis'), vmin=0, vmax=max(vabs))

def plot_data_w_fluid(pXs, pVs, fXs, fVs, sim_info, image_folder, title, L, fix_frame=True, SAVEFIG=True, ex=None, plot_particles=True, plot_fluids=True, side_by_side=False, fluid_plot_type='streamplot', fluid_coloring_scheme='vabs'):
    if not (plot_particles or plot_fluids):
        print('Plotting without anyting to plot. Raising exception...')
        raise RuntimeError
    if side_by_side:
        fig, (axl,axr) = plt.subplots(1,2, figsize=(10,5))
    else:
        fig = plt.figure(figsize=(5,5))
        axl = fig.gca()
        axr = fig.gca()

    if plot_particles:
        plot_points(axl, pXs, pVs)
    if plot_fluids:
        plot_fluid(axr, fXs, fVs, sim_info=sim_info, plot_type = fluid_plot_type, coloring_scheme=fluid_coloring_scheme)

    fig.suptitle(title)
    for ax in (axl,axr):
        plt.sca(ax)
        if fix_frame:
            plt.xlim([sim_info['x_min'],sim_info['x_max']])
            plt.ylim([sim_info['y_min'],sim_info['y_max']])
    plt.tight_layout()

    IMG_NAME=f"{image_folder}/fig{sim_info['time_step_index']:08}.png"
    plt.savefig(IMG_NAME)
    if SAVEFIG:
        ex.add_artifact(IMG_NAME)
    try:
        plt.close(fig)
    except:
        print('Something went wrong with closing the figure')

def plot_mixing(pXs, sim_info, image_folder, title,L,fix_frame,SAVEFIG,ex):
    fig = plt.figure(figsize=(5,5))
    ax = fig.gca()
    
    split = len(pXs)//2

    plt.scatter(pXs[split:,0],pXs[split:,1])
    plt.scatter(pXs[:split,0],pXs[:split,1])

    plt.title(title)
    if fix_frame:
        plt.xlim([sim_info['x_min'],sim_info['x_max']])
        plt.ylim([sim_info['y_min'],sim_info['y_max']])
    plt.tight_layout()

    IMG_NAME=f"{image_folder}/fig{sim_info['time_step_index']:08}.png"
    plt.savefig(IMG_NAME)
    if SAVEFIG:
        ex.add_artifact(IMG_NAME)
    try:
        plt.close(fig)
    except:
        print('Something went wrong with closing the figure')

def generate_video_from_png(image_folder, video_length=10, do_h264 = False):
# Adapted from answer by BoboDarph (Stackoverflow: https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python)
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    if len(images)>0:
        fps = len(images)/video_length
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        video_path=f'{image_folder}/video.avi'
        
        if do_h264:
            video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'H264'), fps, (width,height))
        else:
            video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()
        return video_path
