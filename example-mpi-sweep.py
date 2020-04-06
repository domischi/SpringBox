from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import sys 
import time
from experiment_single import ex

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

for i, vx in enumerate([.1,.2,.5,.8,1.,1.2,1.5,2]):
    if i%size == rank:
        time.sleep(rank) # To mitigate race conditions
        config_updates = {'run_id': i, 'sweep_experiment': True , 'vx': float(vx)}
        ex.run(config_updates=config_updates)
