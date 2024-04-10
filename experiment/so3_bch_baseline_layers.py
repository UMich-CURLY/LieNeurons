import torch
import e3nn

class E3nnMLPBlockNorm(torch.nn.Module):
    def __init__(self, rep_input, rep_output=None) -> None:
        super().__init__()

        self.linear = e3nn.o3.Linear(rep_input, rep_output)
        self.batchnorm = e3nn.nn.BatchNorm(rep_output)
        self.act = e3nn.nn.NormActivation(rep_output, torch.sigmoid)

    def forward(self, x):
        x = self.linear(x)
        x = self.batchnorm(x)
        x = self.act(x)
        return x

# class E3nnMLPBlockS2Grid(torch.nn.Module):
#     def __init__(self, rep_input, l_max, l_max_in=None, grid_size=100) -> None:
#         super().__init__()

#         rep_output = e3nn.io.SphericalTensor(l_max, 1, 1)
#         if rep_input is None:
#             rep_input = e3nn.io.SphericalTensor(l_max_in, 1, 1)

#         self.linear = e3nn.o3.Linear(rep_input, rep_output)
#         self.batchnorm = e3nn.nn.BatchNorm(rep_output)
#         self.act = e3nn.nn.S2Activation(rep_output, torch.sigmoid, grid_size)

#     def forward(self, x):
#         x = self.linear(x)
#         x = self.batchnorm(x)
#         x = self.act(x)
#         return x
    
class E3nnMLPBlockGate(torch.nn.Module):
    def __init__(self, rep_input, irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated) -> None:
        super().__init__()

        rep_hidden = e3nn.o3.Irreps(irreps_scalars) + e3nn.o3.Irreps(irreps_gates) + e3nn.o3.Irreps(irreps_gated)
        self.linear = e3nn.o3.Linear(rep_input, rep_hidden)
        self.batchnorm = e3nn.nn.BatchNorm(rep_hidden)
        self.act = e3nn.nn.Gate(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated)

        self.rep_output = e3nn.o3.Irreps(irreps_scalars) + e3nn.o3.Irreps(irreps_gated)

    def forward(self, x):
        x = self.linear(x)
        x = self.batchnorm(x)
        x = self.act(x)
        return x
    
class E3nnMLPNorm(torch.nn.Module):
    def __init__(self, invariant=False):
        super().__init__()

        rep_input = "2x1e"
        rep_hidden = "1024x1e+1024x2e+1024x3e+1024x4e" # arbitrary a and b in [axbe]
        if invariant:
            rep_output = "1x0e"
        else:
            rep_output = "1x1e"

        self.block1 = E3nnMLPBlockNorm(rep_input, rep_hidden)
        self.block2 = E3nnMLPBlockNorm(rep_hidden, rep_hidden)
        self.block3 = E3nnMLPBlockNorm(rep_hidden, rep_hidden)
        self.block4 = E3nnMLPBlockNorm(rep_hidden, rep_hidden)
        self.block5 = E3nnMLPBlockNorm(rep_hidden, rep_hidden)
        self.block6 = E3nnMLPBlockNorm(rep_hidden, rep_hidden)
        self.out = e3nn.o3.Linear(rep_hidden, rep_output)

    def forward(self, x):
        B, F, _, _ = x.shape
        x = torch.reshape(x, (B, -1))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.out(x)
        return x

# class E3nnMLPS2Grid(torch.nn.Module):
#     def __init__(self, invariant=False):
#         super().__init__()

#         rep_input = "2x1e"
#         rep_hidden_lmax = 20 # arbitrary lmax
#         rep_hidden = e3nn.io.SphericalTensor(rep_hidden_lmax, 1, 1)
#         if invariant:
#             rep_output = "1x0e"
#         else:
#             rep_output = "1x1e"

#         self.block1 = E3nnMLPBlockS2Grid(rep_input, rep_hidden_lmax)
#         self.block2 = E3nnMLPBlockS2Grid(None, rep_hidden_lmax, rep_hidden_lmax)
#         self.block3 = E3nnMLPBlockS2Grid(None, rep_hidden_lmax, rep_hidden_lmax)
#         self.block4 = E3nnMLPBlockS2Grid(None, rep_hidden_lmax, rep_hidden_lmax)
#         self.block5 = E3nnMLPBlockS2Grid(None, rep_hidden_lmax, rep_hidden_lmax)
#         self.block6 = E3nnMLPBlockS2Grid(None, rep_hidden_lmax, rep_hidden_lmax)
#         self.out = e3nn.o3.Linear(rep_hidden, rep_output)

#     def forward(self, x):
#         B, F, _, _ = x.shape
#         x = torch.reshape(x, (B, -1))
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         x = self.block5(x)
#         x = self.block6(x)
#         x = self.out(x)
#         return x
    
class E3nnMLPGate(torch.nn.Module):
    def __init__(self, invariant=False):
        super().__init__()

        rep_input = "2x1e"
        rep_hidden_scalars = "16x0e"    # arbitrary a in [ax0e]
        rep_hidden_gates = "32x0e"      # arbitrary b in [bx0e]
        # arbitrary ci and di in [ci x di e] and there can be multiple such terms, but \sum ci + \sum di = b
        rep_hidden_gated = "16x1e+16x2e"
        rep_hidden = e3nn.o3.Irreps(rep_hidden_scalars) + e3nn.o3.Irreps(rep_hidden_gated)
        if invariant:
            rep_output = "1x0e"
        else:
            rep_output = "1x1e"

        self.block1 = E3nnMLPBlockGate(rep_input, rep_hidden_scalars, [torch.tanh], rep_hidden_gates, [torch.tanh], rep_hidden_gated)
        self.block2 = E3nnMLPBlockGate(rep_hidden, rep_hidden_scalars, [torch.tanh], rep_hidden_gates, [torch.tanh], rep_hidden_gated)
        self.out = e3nn.o3.Linear(rep_hidden_gated, rep_output)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.out(x)
        return x

class EquivariantMLPDraft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # linear layer
        rep_input = "2x1e"

        rep_hidden = "16x0e+32x0e+16x1e+16x2e"    # for gate activation
        rep_hidden_after_act = "16x1e+16x2e"    # for gate activation
        # rep_hidden = "1x0e+3x1e+5x2e+7x3e"  # for s2 activation
        # rep_hidden_after_act = rep_hidden   # for s2 activation
        # rep_hidden = "16x1e+16x2e+16x3e+16x4e"    # for norm activation
        # rep_hidden_after_act = rep_hidden         # for norm activation

        rep_output_equiv = "1x1e"
        rep_output_inv = "1x0e"

        self.linear = e3nn.o3.Linear(rep_input, rep_hidden)
        self.batchnorm = e3nn.nn.BatchNorm(rep_hidden)
        self.act = e3nn.nn.Gate("16x0e", [torch.tanh], "32x0e", [torch.tanh], "16x1e+16x2e")
        # self.act = e3nn.nn.NormActivation(rep_hidden, torch.sigmoid)
        # self.act = e3nn.nn.S2Activation(rep_hidden, torch.sigmoid, 100)
        # self.act = e3nn.nn.SO3Activation(3, 3, torch.sigmoid, 100)

        self.out = e3nn.o3.Linear(rep_hidden_after_act, rep_output_inv)     # invariant output
        # self.out = e3nn.o3.Linear(rep_hidden_after_act, rep_output_equiv)   # equivariant output

    def forward(self, x):
        x = self.linear(x)
        x = self.batchnorm(x)
        x = self.act(x)
        x = self.out(x)
        return x
    

class E3nnMLPBlockS2Grid(torch.nn.Module):
    def __init__(self, rep_input, l_max, l_max_in=None, multiplicity=3, grid_size=100) -> None:
        super().__init__()
        # rep_output = (e3nn.io.SphericalTensor(l_max, 1, 1) * multiplicity).simplify()
        # if rep_input is None:
        #     rep_input = (e3nn.io.SphericalTensor(l_max_in, 1, 1) * multiplicity).simplify()
        rep_output = "128x0e+128x1e+128x2e+128x3e+128x4e+128x5e" #e3nn.io.SphericalTensor(l_max, 1, 1)
        if rep_input is None:
            # rep_input = e3nn.io.SphericalTensor(l_max_in, 1, 1)
            rep_input = "128x0e+128x1e+128x2e+128x3e+128x4e+128x5e"
        rep_output_hidden = "1x0e+1x1e+1x2e+1x3e+1x4e+1x5e"
        self.linear = e3nn.o3.Linear(rep_input, rep_output)
        self.batchnorm = e3nn.nn.BatchNorm(rep_output)
        self.linear2 = e3nn.o3.Linear(rep_output, rep_output_hidden)
        self.act = e3nn.nn.S2Activation(rep_output_hidden, torch.sigmoid, grid_size)
        self.linear3 = e3nn.o3.Linear(rep_output_hidden, rep_output)
    def forward(self, x):
        x = self.linear(x)
        x = self.batchnorm(x)
        x = self.linear2(x)
        x = self.act(x)
        x = self.linear3(x)
        return x

class E3nnMLPS2Grid(torch.nn.Module):
    def __init__(self, invariant=False):
        super().__init__()
        rep_input = "2x1e"
        rep_hidden_lmax = 5 # arbitrary lmax
        # rep_hidden = e3nn.io.SphericalTensor(rep_hidden_lmax, 1, 1)
        rep_hidden = "128x0e+128x1e+128x2e+128x3e+128x4e+128x5e"
        if invariant:
            rep_output = "1x0e"
        else:
            rep_output = "1x1e"
        self.block1 = E3nnMLPBlockS2Grid(rep_input, rep_hidden_lmax)
        self.block2 = E3nnMLPBlockS2Grid(None, rep_hidden_lmax, rep_hidden_lmax)
        self.block3 = E3nnMLPBlockS2Grid(None, rep_hidden_lmax, rep_hidden_lmax)
        self.block4 = E3nnMLPBlockS2Grid(None, rep_hidden_lmax, rep_hidden_lmax)
        self.out = e3nn.o3.Linear(rep_hidden, rep_output)
    def forward(self, x):
        B, F, _, _ = x.shape
        x = torch.reshape(x, (B, -1))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.out(x)
        return x


if __name__ == "__main__":
    # network = EquivariantMLPDraft()

    ########## USE ONE OF THE FOLLOWING NETWORKS ##########
    # network = E3nnMLPGate(invariant=True)
    # network = E3nnMLPS2Grid(invariant=True)
    network = E3nnMLPNorm(invariant=True)

    # create a random input
    x = torch.randn(100, 6)

    # apply the network
    y = network(x)
    print(y.shape)