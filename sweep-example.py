from example import ex
import numpy as np
from tqdm import tqdm

cutoffs = [0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
for c in tqdm(cutoffs):
    ex.run(config_updates={'cutoff': c/4,'lower_cutoff': (c/25)/4}) # The /4 because of the convention conversion between Matt's code and my code
