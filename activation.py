import numba
import numpy as np

@numba.njit(parallel=True)
def rectangle_activation(ps, AR):
    return (ps[:,0]>-1) * (ps[:,0]<1) * (ps[:,1]>-AR) * (ps[:,1]<AR)


def activation_fn_dispatcher(_config, t):
    if _config['activation_fn_type'] == 'const-rectangle':
        return lambda ps: rectangle_activation(ps, _config['AR'])
    else:
        raise RuntimeError('Unrecognized activation_fn_type')
