# Adapted from answer by BoboDarph (Stackoverflow: https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python)

import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

def plot_fluid(fXs, fVs,i,  image_folder, title,L,fix_frame=True, SAVEFIG=True, ex=None):
    fig=plt.figure(figsize=(10,10))
    vabs = np.linalg.norm(fVs,axis=1)
    X = np.linspace(min(fXs.flatten()),max(fXs.flatten()),int(np.sqrt(len(fXs)))) ## a bit of a hack, but it works for now
    X,Y = np.meshgrid(X,X)
    qc=plt.quiver(X,Y, fVs[:,0],fVs[:,1], vabs , units='xy', pivot='mid', scale=.1)
    plt.title(title)
    if fix_frame:
        plt.xlim([-L,L])
        plt.ylim([-L,L])
    IMG_NAME=f'{image_folder}/fluid{i:08}.png'
    plt.savefig(IMG_NAME)
    if SAVEFIG:
        ex.add_artifact(IMG_NAME)
    try:
        plt.close(fig)
    except:
        print('Something went wrong with closing the figure')

def plot_points(pXs, pVs, i,cutoff,lower_cutoff, image_folder, title, AR,L, fix_frame=True, SAVEFIG=True, ex=None):
    fig=plt.figure(figsize=(12,10))
    vabs = np.linalg.norm(pVs, axis=1)
    sc=plt.scatter(pXs[:,0],pXs[:,1], c=vabs, cmap=plt.get_cmap('viridis'), vmin=0, vmax=max(vabs))
    plt.colorbar(sc)
    plt.title(title)
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

def generate_video_from_png(image_folder):
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
