import autograd.numpy as np
import numpy as nump
from autograd import grad
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg
from autograd.misc.optimizers import adam

def f(x):
    xsum = x.sum(axis=0);
    return np.sign(xsum)
    # return 0.05 * xsum**2 + np.sin(xsum);

dim       = 1
num_train = 66
sn        = 1e-2
train_x   = 5 * np.random.randn(dim, num_train)
train_y   = f(train_x) + sn * np.random.randn(1, num_train)
m         = 80

num_test = 1000
test_x   = np.linspace(-50, 50, num_test).reshape(dim, num_test);
test_y   = f(test_x);

def relu(x):
    return np.maximum(x, 0.0)

def oneLayerNN(x, w):
    dim, num_data = x.shape
    bias          = np.ones(num_data);
    xx            = np.array([bias, x[0, :]])
    ww            = w.reshape(dim+1, m)
    transformed   = np.dot(ww.T, xx)
    return np.tanh(transformed)



def chol_solve(L, y):
    """
    K = L.dot(L.T)
    return inv(K) * y
    """
    v = np.linalg.solve(L, y)
    return np.linalg.solve(L.T, v)


def chol_inv(L):
    return chol_solve(L, np.eye(L.shape[0]))

def loss(w):
    sn2 = sn ** 2
    Phi = oneLayerNN(train_x, w)
    Qmm = np.dot(Phi, Phi.T)
    A   = sn2 * np.eye(m) + Qmm
    LA  = np.linalg.cholesky(A)

    data_fit_1 = np.dot(train_y, train_y.T)
    data_fit_2 = np.dot(Phi, train_y.T)
    data_fit_2 = chol_solve(LA, data_fit_2)
    data_fit_2 = np.dot(Phi.T, data_fit_2)
    data_fit_2 = np.dot(train_y, data_fit_2)

    s, logDetA       = np.linalg.slogdet(A);
    model_complexity = (num_train - m) * np.log(sn2) + logDetA
    loss             = model_complexity + (data_fit_1 - data_fit_2) / sn2
    print(loss)
    return loss


def predict(w, test_x):
    sn2 = sn ** 2
    Phi = oneLayerNN(train_x, w)
    Qmm = np.dot(Phi, Phi.T)
    A   = sn2 * np.eye(m) + Qmm
    LA  = np.linalg.cholesky(A)


    alpha = train_y.copy()
    alpha = Phi.dot(alpha.T)
    alpha = chol_solve(LA, alpha)
    alpha = Phi.T.dot(alpha)
    alpha = train_y.T - alpha;
    alpha = alpha / sn2;
    alpha = Phi.dot(alpha)

    Phi_test = oneLayerNN(test_x, w)
    py       = Phi_test.T.dot(alpha)
    return py

gloss  = grad(loss)
print("Start")
w0     = np.random.randn((dim+1) * m)
loss0  = loss(w0)
gloss0 = gloss(w0)

# epsi = 1e-3
# for i in range(w0.size):
#     w1     = np.copy(w0)
#     w2     = np.copy(w0)
#     w1[i] += epsi
#     w2[i] -= epsi
#     gdiff = (loss(w1) - loss(w2))/(2*epsi)
#     print([gdiff[0][0], gloss0[i]])


py0     = predict(w0, test_x)
best_w  = fmin_cg(loss, w0, gloss, maxiter=100)
py_best = predict(best_w, test_x);

print(py_best.shape)
print(test_x.shape)
print(test_y.shape)

plt.plot(test_x[0], py_best.T[0])
plt.plot(test_x[0], test_y.reshape(1, num_test)[0])
plt.show()


np.savetxt("train_x", train_x.T)
np.savetxt("train_y", train_y)
np.savetxt("test_x", test_x.T)
np.savetxt("test_y", test_y)
np.savetxt("pred_y_init", py0)
np.savetxt("pred_y", py_best)
