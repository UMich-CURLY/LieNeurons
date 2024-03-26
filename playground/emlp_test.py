import torch

import emlp.nn.pytorch as nn
from emlp.reps import T,V
from emlp.groups import SO



if __name__ == '__main__':
    # Define the input representation
    G = SO(3)
    reps = V(G)
    reps_out = V(G)

    Q = (reps>>reps_out).equivariant_basis() 
    print(f"Basis matrix of shape {Q.shape}")
    # Define the output representation