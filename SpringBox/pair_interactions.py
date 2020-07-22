import numba
import numpy as np
from scipy.spatial.distance import pdist, squareform


def active_particles(pXs, prv_acc, activation_fn, _config):
    # Particles that were previously active, now can loose their activation with a rate of activation_decay_rate
    prv_acc = prv_acc * (np.random.rand(len(prv_acc)) > _config['activation_decay_rate'] * _config['dt'])
    # Or particles can become active when they enter the light activated area
    new_acc = activation_fn(pXs)
    return np.maximum(prv_acc, new_acc)


@numba.jit
def spring_forces(acc, pXs, Dij, dpXs, _config):
    rhs = np.zeros_like(pXs)
    n_part = _config['n_part']
    k = _config['spring_k']
    r0 = _config['spring_r0']
    Iij = Dij * np.outer(acc, acc)
    Iij = (Iij > _config['spring_lower_cutoff']) * (Iij < _config['spring_cutoff'])
    for i in range(n_part):
        for j in range(i + 1, n_part):
            if Iij[i, j] != 0:
                rhs[i] += -k * ((Dij[i, j] - r0) / Dij[i, j]) * dpXs[i][j]
                rhs[j] += +k * ((Dij[i, j] - r0) / Dij[i, j]) * dpXs[i][j]
    return rhs


@numba.jit
def LJ_forces(acc, pXs, Dij, dpXs, _config):
    n_part = _config['n_part']
    rhs = np.zeros_like(pXs)
    eps = _config['LJ_eps']
    r0 = _config['LJ_r0']
    r0_6 = r0 ** 6
    LJ_pre = 12 * eps * r0_6
    Dij6 = Dij ** 6
    np.clip(Dij6, 1e-6, None, out=Dij6)
    Iij = (Dij < _config['LJ_cutoff']) * np.outer(acc, acc)
    for i in range(n_part):
        for j in range(i + 1, n_part):
            if Iij[i, j] != 0:
                rhs[i] += -LJ_pre * dpXs[i][j] * ((Dij6[i, j] - r0_6) / (Dij[i, j] * Dij6[i, j]))
                rhs[j] += +LJ_pre * dpXs[i][j] * ((Dij6[i, j] - r0_6) / (Dij[i, j] * Dij6[i, j]))
    return rhs


@numba.jit
def periodic_dist(u, v, length):
    x_diff = min(abs(u[0] - v[0]), 2 * length - abs(u[0] - v[0]))
    y_diff = min(abs(u[1] - v[1]), 2 * length - abs(u[1] - v[1]))
    return np.sqrt(x_diff ** 2 + y_diff ** 2)


def periodic_distance_vectors(pXs, L):
    diffs = np.array([np.subtract.outer(p, p) for p in pXs.T]).T
    return np.remainder(diffs + L, 2 * L) - L


@numba.jit
def RHS(pXs, prv_acc, activation_fn, _config):
    rhs = np.zeros_like(pXs)
    acc = active_particles(pXs, prv_acc, activation_fn, _config)
    length = _config['L']
    if _config.get('periodic_boundary', False):
        Dij = squareform(pdist(pXs, metric=periodic_dist, length=length))
        dpXs = periodic_distance_vectors(pXs, length)
    else:
        Dij = squareform(pdist(pXs))
        dpXs = pXs[:, None] - pXs
    np.clip(Dij, 1e-6, None, out=Dij)
    assert (_config['spring_k'] > 0 or _config['LJ_eps'] > 0)
    ## Spring
    if _config['spring_k'] > 0:
        rhs += spring_forces(acc, pXs, Dij, dpXs, _config)
    ## Lennard-Jones
    if _config['LJ_eps'] > 0:
        rhs += LJ_forces(acc, pXs, Dij, dpXs, _config)
    return rhs, acc
