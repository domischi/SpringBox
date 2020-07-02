import numpy as np

def rectangle_activation(ps, AR):
    return (ps[:,0]>-1) * (ps[:,0]<1) * (ps[:,1]>-AR) * (ps[:,1]<AR)

def moving_circle_activation(ps, x_center, R):
    return (np.linalg.norm(ps-x_center,axis=1)<R)

def activation_pattern(ps, X, Y, A):
    ind_x = np.vectorize(lambda a: np.argmax(a<X)-1)(ps[:,0])
    ind_y = np.vectorize(lambda a: np.argmax(a<Y)-1)(ps[:,1])
    return A[ind_x,ind_y]


def activation_fn_dispatcher(_config, t, **kwargs):
    if   _config['activation_fn_type'] == 'const-rectangle':
        return lambda ps: rectangle_activation(ps, _config['AR'])
    elif _config['activation_fn_type'] == 'moving-circle':
        return lambda ps: moving_circle_activation(ps,_config['v_circ']*t+_config['x_0_circ'],_config['activation_circle_radius'])
    elif _config['activation_fn_type'] == 'activation_matrix':
        assert('lx' in kwargs)
        assert('ly' in kwargs)
        assert('lh' in kwargs)

        return lambda ps: activation_pattern(ps, kwargs['lx'], kwargs['ly'], kwargs['lh'])
    else:
        raise RuntimeError('Unrecognized activation_fn_type')
