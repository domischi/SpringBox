import h5py
import numpy as np

def print_group(f, group_name, indent=''):
    view = f[group_name]
    for k in view.keys():
        if isinstance(view[k], h5py.Group):
            print_group(view, k, indent='  ')
        else:
            print(f"{indent}{group_name}/{k}: {type(view[k])}")


with h5py.File('data_dump.h5', 'r') as f:
    print(f.keys())
    for iteration_group in f.keys():
        print_group(f, iteration_group)
