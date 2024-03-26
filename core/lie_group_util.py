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


# def bacth_exp_so3(A):
#   # A (B,3,3)
#   I = torch.eye(3).to(A.device)
#   theta = torch.sqrt(-torch.trace(torch.matmul(A,A))/2.0)
#   return I + (torch.sin(theta)/theta)*A + ((1-torch.cos(theta))/(theta*theta))*torch.matmul(A,A)

def batch_trace(A):
  return A.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

def exp_so3(A):
  I = torch.eye(3).to(A.device)
  # print("A: ",A.shape)
  theta = torch.sqrt(-batch_trace(torch.matmul(A,A))/2.0).reshape(-1,1,1)
  # print("theta: ", theta.shape)
  return I + (torch.sin(theta)/theta)*A + ((1-torch.cos(theta))/(theta**2))*torch.matmul(A,A)

def log_SO3(R):

  # print("trace: ", torch.trace(R))
  print("R",R)
  print("trace",(batch_trace(R)-1)/2.0)
  theta = torch.acos((batch_trace(R)-1)/2.0)
  theta2 = torch.asin(torch.sqrt((3-batch_trace(R))*(1+batch_trace(R)))/2.0)
  print("theta: ", theta)
  print("theta2: ", theta2)


  # if torch.isnan(theta):
  #   return torch.zeros((3,3)).to(R.device)
  # print("theta: ", theta)
  # if theta - np.pi < 1e-6:
  #   return theta/2/(np.pi-theta)*(R-R.T)
  # elif theta > np.pi:
  #   theta = np.pi-theta
  # K = (theta/(2*torch.sin(theta)))[:,None,None]*(R-R.transpose(-1,-2))

  return (theta/(2*torch.sin(theta)))[:,None,None]*(R-R.transpose(-1,-2))


def BCH_first_order_approx(X,Y):
  return X+Y

def BCH_second_order_approx(X,Y):
  return X+Y+1/2*lie_bracket(X,Y)

def BCH_third_order_approx(X,Y):
  return X+Y+1/2*lie_bracket(X,Y)+1/12*lie_bracket(X,lie_bracket(X,Y))-1/12*lie_bracket(Y,lie_bracket(X,Y))

def BCH_approx(X,Y):
  return X+Y+1/2*lie_bracket(X,Y)+1/12*lie_bracket(X,lie_bracket(X,Y))-1/12*lie_bracket(Y,lie_bracket(X,Y))\
            -1/24*lie_bracket(Y,lie_bracket(X,lie_bracket(X,Y)))-1/720*lie_bracket(Y,lie_bracket(Y,lie_bracket(Y,lie_bracket(Y,X))))\
            -1/720*lie_bracket(X,lie_bracket(X,lie_bracket(X,lie_bracket(X,Y))))

def BCH_so3(X,Y):
  x = vee_so3(X)
  y = vee_so3(Y)
  theta = torch.norm(x)
  phi = torch.norm(y)          
  delta = torch.acos(x.transpose(-1,0)@y/torch.norm(x)/torch.norm(y)) # angles between x and y
  a = torch.sin(theta)*torch.cos(phi/2.0)*torch.cos(phi/2.0)-torch.sin(phi)*torch.sin(theta/2.0)*torch.sin(theta/2.0)*torch.cos(delta)
  b = torch.sin(phi)*torch.cos(theta/2.0)*torch.cos(theta/2.0)-torch.sin(theta)*torch.sin(phi/2.0)*torch.sin(phi/2.0)*torch.cos(delta)
  c = 1/2.0*torch.sin(theta)*torch.sin(phi)-2.0*torch.sin(theta/2.0)*torch.sin(theta/2.0)*torch.sin(phi/2.0)*torch.sin(phi/2.0)*torch.cos(delta)
  d = torch.sqrt(a*a+b*b+2.0*a*b*torch.cos(delta)+c*c*torch.sin(delta)*torch.sin(delta))

  alpha = torch.asin(d)*a/d/theta
  beta = torch.asin(d)*b/d/phi
  gamma = torch.asin(d)*c/d/theta/phi

  return alpha*X+beta*Y+gamma*lie_bracket(X,Y)