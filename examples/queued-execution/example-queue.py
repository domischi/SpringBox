import sys
import time
from example import ex

for i, vx in enumerate([.1,.2,.5,.8,1.,1.2,1.5,2]):
        config_updates = {'run_id': i, 'sweep_experiment': True , 'vx': float(vx)}
        ex.run(config_updates=config_updates, options={'--queue': True})
