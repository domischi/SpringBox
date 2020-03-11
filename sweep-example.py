from example import ex
import numpy as np
from tqdm import tqdm
import itertools
from multiprocessing import Pool

tmp1 = np.linspace(.1,1.,5)
tmp2 = np.linspace(.9,1,3)
ARs=np.union1d(tmp1,tmp2)[::-1]
del tmp1, tmp2

drag_radii = [.01]

space = itertools.product(drag_radii,ARs)

def f(x):
    config_updates = {'run_id': x[0], 'AR': x[1][1], 'Rdrag': x[1][0]}
    assert(config_updates['AR']>0)
    ex.run(config_updates=config_updates)

p = Pool(3)
p.map(f, enumerate(space))
