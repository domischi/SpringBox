import numpy as np
import os
import imageio
import glob

def inactive_activation_fn(ps):
    return np.zeros(len(ps)).astype(np.byte)

def rectangle_activation(ps, AR):
    return ((ps[:,0]>-1) * (ps[:,0]<1) * (ps[:,1]>-AR) * (ps[:,1]<AR)).astype(np.byte)

def moving_circle_activation(ps, x_center, R):
    return (np.linalg.norm(ps-x_center,axis=1)<R).astype(np.byte)

def dumbbell_activation(ps, R, l, w):
    # R is the radius of each dumbbell circle
    # l is the distance from the center of the dumbbell to the center of the circle
    # w is the width of the bar
    part_of_bar = (ps[:,0]>-l) * (ps[:,0]<l) * (ps[:,1]>-w/2) * (ps[:,1]<w/2)
    part_of_l_circle = np.linalg.norm(ps-np.array([l,0]), axis=1)<R
    part_of_r_circle = np.linalg.norm(ps+np.array([l,0]), axis=1)<R
    return (np.logical_or(part_of_bar, np.logical_or(part_of_l_circle, part_of_r_circle))).astype(np.byte)


def activation_pattern(ps, X, Y, A):
    ind_x = np.vectorize(lambda a: np.argmax(a<X)-1)(ps[:,0])
    ind_y = np.vectorize(lambda a: np.argmax(a<Y)-1)(ps[:,1])
    ret_val = A[ind_x,ind_y]
    assert(min(ret_val)>=-1 and max(ret_val)<=1)
    return ret_val

def rgb2gray(rgb):
    if len(rgb.shape)<2:
        raise RuntimeError(f"Unable to convert image of dimensions {rgb.shape} to greyscale.")
    elif len(rgb.shape)==2:
        return rgb
    else:
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def activation_pattern_from_image_path(ps, fname, L, ex):
    ex.add_resource(fname)
    activation_mat = imageio.imread(fname) # Inefficient as it is done in every timestep, but specifically with eventual video hardly avoidable
    activation_mat = rgb2gray(activation_mat) # Make it gray scale
    if np.max(activation_mat)<20:
        raise RuntimeWarning(f'max value of activation_matrix is below 20, namely {np.max(activation_mat)=}')
    else:
        activation_mat = activation_mat/255. # normalize it properly
    activation_mat = (activation_mat>0.5).astype('int8') # binarzie
    activation_mat = np.rot90(activation_mat, k=3) # rotate the same way up as input
    nx = activation_mat.shape[0]
    X = np.linspace(-L, L, num=nx, endpoint=False) ## Assume square image
    return activation_pattern(ps, X, X, activation_mat)

def activation_pattern_from_video(ps, fpath, t, T, L, ex):
    flist = sorted(glob.glob(fpath))
    index = min(int(len(flist)*t/T), len(flist)-1)
    return activation_pattern_from_image_path(ps, flist[index], L, ex)


def activation_fn_dispatcher(_config, t, **kwargs):
    if   _config['activation_fn_type'] == 'const-rectangle':
        return lambda ps: rectangle_activation(ps, _config['AR'])
    elif _config['activation_fn_type'] == 'moving-circle':
        return lambda ps: moving_circle_activation(ps,np.array(_config['v_circ'])*t+np.array(_config['x_0_circ']),_config['activation_circle_radius'])
    elif _config['activation_fn_type'] == 'dumbbell':
        return lambda ps: dumbbell_activation(ps, _config['activation_circle_radius'], _config['dumbbell_dist'], _config['dumbbell_width'])
    elif _config['activation_fn_type'] == 'activation_matrix':
        assert('lx' in kwargs)
        assert('ly' in kwargs)
        assert('lh' in kwargs)
        return lambda ps: activation_pattern(ps, kwargs['lx'], kwargs['ly'], kwargs['lh'])
    elif _config['activation_fn_type'] == 'image':
        fpath = _config['activation_image_filepath']
        assert(os.path.exists(fpath))
        return lambda ps: activation_pattern_from_image_path(ps, fpath, _config["L"], kwargs['experiment'])
    elif _config['activation_fn_type'] == 'video':
        fpath = _config['activation_image_filepath']
        return lambda ps: activation_pattern_from_video(ps, fpath, t=t, T=_config["T"], L=_config["L"], ex=kwargs['experiment'])
    else:
        raise RuntimeError('Unrecognized activation_fn_type')
