import sys
import numpy as np
import torch
sys.path.append('.')


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from core.lie_alg_util import lie_bracket, vee_so3

def skew(v):
    # M = [[0 , -v[2], v[1]],
    #     [v[2], 0, -v[0]],
    #     [-v[1], v[0], 0]]

    v2 = v.clone()
    M = torch.zeros((3,3))
    M[0,1] = -v2[2]
    M[0,2] = v2[1]
    M[1,0] = v2[2]
    M[1,2] = -v2[0]
    M[2,0] = -v2[1]
    M[2,1] = v2[0]
    print("M", M)
    return M


def exp_hat_and_so3(w):
  I = torch.eye(3)
  theta = torch.norm(w)
  A = skew(w)
  return I + (torch.sin(theta)/theta)*A + ((1-torch.cos(theta))/(theta*theta))*torch.matmul(A,A)

def exp_so3(A):
  I = torch.eye(3)
  theta = torch.sqrt(-torch.trace(A@A)/2.0)
  return I + (torch.sin(theta)/theta)*A + ((1-torch.cos(theta))/(theta*theta))*torch.matmul(A,A)

def log_SO3(R):
  theta = torch.acos((torch.trace(R)-1)/2.0)
  return (theta/(2*torch.sin(theta)))*(R-R.T)

def BCH_approx(X,Y):
  return X+Y+1/2*lie_bracket(X,Y)+1/12*lie_bracket(X,lie_bracket(X,Y))-1/12*lie_bracket(Y,lie_bracket(X,Y))\
            -1/24*lie_bracket(Y,lie_bracket(X,lie_bracket(X,Y)))-1/720*lie_bracket(Y,lie_bracket(Y,lie_bracket(Y,lie_bracket(Y,X))))\
            -1/720*lie_bracket(X,lie_bracket(X,lie_bracket(X,lie_bracket(X,Y))))

def BCH_so3(X,Y):
  x = vee_so3(X)
  y = vee_so3(Y)
  theta = torch.sqrt(-torch.trace(X@X)/2.0)
  phi = torch.sqrt(-torch.trace(Y@Y)/2.0)             
  delta = torch.acos(-torch.trace(X@Y)/2.0)/theta/phi # angle between X and Y
  # delta = torch.acos(x.T@y/torch.norm(x)/torch.norm(y))
  a = torch.sin(theta)*torch.cos(phi/2.0)*torch.cos(phi/2.0)-torch.sin(phi)*torch.sin(theta/2.0)*torch.sin(theta/2.0)*torch.cos(delta)
  b = torch.sin(phi)*torch.cos(theta/2.0)*torch.cos(theta/2.0)-torch.sin(theta)*torch.sin(phi/2.0)*torch.sin(phi/2.0)*torch.cos(delta)
  c = 1/2.0*torch.sin(theta)*torch.sin(phi)-2.0*torch.sin(theta/2.0)*torch.sin(theta/2.0)*torch.sin(phi/2.0)*torch.sin(phi/2.0)*torch.cos(delta)
  d = torch.sqrt(a*a+b*b+2.0*a*b*torch.cos(delta)+c*c*torch.sin(delta)*torch.sin(delta))

  alpha = torch.asin(d)*a/d/theta
  beta = torch.asin(d)*b/d/phi
  gamma = torch.asin(d)*c/d/theta/phi

  return alpha*X+beta*Y+gamma*lie_bracket(X,Y)