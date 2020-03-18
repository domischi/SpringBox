import numpy as np

def rectangle_activation(ps, AR):
    return (ps[:,0]>-1) * (ps[:,0]<1) * (ps[:,1]>-AR) * (ps[:,1]<AR)

def moving_circle_activation(ps, x_center, R):
    return (np.linalg.norm(ps-x_center,axis=1)<R)


def activation_fn_dispatcher(_config, t):
    if   _config['activation_fn_type'] == 'const-rectangle':
        return lambda ps: rectangle_activation(ps, _config['AR'])
    elif _config['activation_fn_type'] == 'moving-circle':
        return lambda ps: moving_circle_activation(ps,_config['v_circ']*t+_config['x_0_circ'],_config['activation_circle_radius'])
    else:
        raise RuntimeError('Unrecognized activation_fn_type')
