import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

def plot_fluid(ax, fXs, fVs, plot_type='streamplot'):
    plt.sca(ax)
    vabs = np.linalg.norm(fVs,axis=1)
    plt.title('Fluids')
    ng = int(np.sqrt(len(fXs)))
    X = np.linspace(min(fXs.flatten()),max(fXs.flatten()), ng) ## a bit of a hack, but it works for now
    if plot_type == 'quiver':
        X,Y = np.meshgrid(X,X)
        p=plt.quiver(X,Y, fVs[:,0],fVs[:,1], vabs , units='xy', pivot='mid')
    elif plot_type == 'streamplot':
        p=plt.streamplot(X,X, fVs[:,0].reshape(ng,ng),fVs[:,1].reshape(ng,ng), color=vabs.reshape(ng,ng))
    else:
        print('Illegal type of fluid plot. Check the code.')

def plot_points(ax, pXs, pVs):
    plt.sca(ax)
    plt.title('Particles')
    vabs = np.linalg.norm(pVs, axis=1)
    sc=plt.scatter(pXs[:,0],pXs[:,1], c=vabs, cmap=plt.get_cmap('viridis'), vmin=0, vmax=max(vabs))

def plot_data(pXs, pVs, fXs, fVs, i, image_folder, title, L, fix_frame=True, SAVEFIG=True, ex=None, plot_particles=True, plot_fluids=True, side_by_side=False):
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
        plot_fluid(axr, fXs, fVs)

    fig.suptitle(title)
    if fix_frame:
        plt.xlim([-L,L])
        plt.ylim([-L,L])
    plt.tight_layout()

    IMG_NAME=f'{image_folder}/fig{i:08}.png'
    plt.savefig(IMG_NAME)
    if SAVEFIG:
        ex.add_artifact(IMG_NAME)
    try:
        plt.close(fig)
    except:
        print('Something went wrong with closing the figure')

def generate_video_from_png(image_folder):
# Adapted from answer by BoboDarph (Stackoverflow: https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python)
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    if len(images)>0:
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        video_path=f'{image_folder}/video.avi'

        video = cv2.VideoWriter(video_path, 0, 30, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()
        return video_path
