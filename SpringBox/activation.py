import numpy as np

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
    if min(ret_val)<-1 or max(ret_val)>1:
        print('-'*80)
        print(f'activation_pattern min(A): {min(A)}')
        print(f'activation_pattern max(A): {max(A)}')
        print(f'activation_pattern min(R): {min(ret_val)}')
        print(f'activation_pattern max(R): {max(ret_val)}')
        print(f'activation_pattern min(ind_x): {min(ind_x)}')
        print(f'activation_pattern max(ind_x): {max(ind_x)}')
        print(f'activation_pattern min(ind_y): {min(ind_y)}')
        print(f'activation_pattern max(ind_y): {max(ind_y)}')
        print('-'*80)
    return ret_val


def activation_fn_dispatcher(_config, t, **kwargs):
    if   _config['activation_fn_type'] == 'const-rectangle':
        return lambda ps: rectangle_activation(ps, _config['AR'])
    elif _config['activation_fn_type'] == 'moving-circle':
        return lambda ps: moving_circle_activation(ps,_config['v_circ']*t+_config['x_0_circ'],_config['activation_circle_radius'])
    elif _config['activation_fn_type'] == 'dumbbell':
        return lambda ps: dumbbell_activation(ps, _config['activation_circle_radius'], _config['dumbbell_dist'], _config['dumbbell_width'])
    elif _config['activation_fn_type'] == 'activation_matrix':
        assert('lx' in kwargs)
        assert('ly' in kwargs)
        assert('lh' in kwargs)

        return lambda ps: activation_pattern(ps, kwargs['lx'], kwargs['ly'], kwargs['lh'])
    else:
        raise RuntimeError('Unrecognized activation_fn_type')
