import numpy as np

w0 = np.random.rand(3)
a0 = np.random.rand(3)

def hat(x: np.ndarray):
    if x.shape == (3,1):
        x = x.flatten()
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def SO3_exp(w):
    theta = np.linalg.norm(w)
    w = w/theta
    output = np.eye(3) + np.sin(theta)*hat(w) + (1-np.cos(theta))*hat(w)@hat(w)
    return output


def f_1(X):
    R = X[0:3,0:3]
    v = X[0:3,3]
    p = X[0:3,4]
    d = X[0:3,5]
    output = np.eye(6)
    output[0:3,0:3] = R @ hat(w0)
    output[0:3,3] = R @ a0 + np.array([0,0,9.8])
    output[0:3,4] = v

    return output

def X_rand():
    output = np.eye(6)
    output[0:3,0:3] = SO3_exp(np.random.rand(3))
    output[0:3,3] = np.random.rand(3)
    output[0:3,4] = np.random.rand(3)
    output[0:3,5] = np.random.rand(3)
    return output

X1 = X_rand()
X2 = X_rand()

error = f_1(X1@X2) - (f_1(X1)@X2 + X1@f_1(X2)-X1@f_1(np.eye(6))@X2)
print("norm of error:", np.linalg.norm(error))

## 

def f_2(X):
    R = X[0:3,0:3]
    v = X[0:3,3]
    output = np.eye(4)
    output[0:3,0:3] = R @ hat(w0)
    output[0:3,3] = -hat(w0) @ v + a0 + R @ np.array([0,0,9.8])
    return output

def X_rand_2():
    output = np.eye(4)
    output[0:3,0:3] = SO3_exp(np.random.rand(3))
    output[0:3,3] = np.random.rand(3)
    return output

X1 = X_rand_2()
X2 = X_rand_2()

error = f_2(X1@X2) - (f_2(X1)@X2 + X1@f_2(X2)-X1@f_2(np.eye(4))@X2)
print("norm of error:", np.linalg.norm(error))
