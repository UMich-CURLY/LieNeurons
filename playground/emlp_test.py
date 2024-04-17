import torch

import emlp.nn.pytorch as nn
from emlp.reps import T,V,sparsify_basis
from emlp.groups import SO, SL



if __name__ == '__main__':
    # Define the input representation
    G = SL(3)
    reps = V(G)
    reps_out = (V**1*V.T**1)(G)
    
    print(reps.rho(G))
    print(reps.size())
    print(reps_out.size())
    Q = (reps>>reps_out).equivariant_basis() 
    print(f"Basis matrix of shape {Q.shape}")
    # print(sparsify_basis(Q).reshape(3,3))
    print(reps(G).size())
    # Define the output representation