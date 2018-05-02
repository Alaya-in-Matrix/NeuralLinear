import autograd.numpy as np
from autograd import grad
import autograd.numpy.random as npr
from scipy.optimize import fmin_cg, fmin_l_bfgs_b
import matplotlib.pyplot as plt
import math
import sys

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(x, 0.0)

def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

def erf(x):
    # save the sign of x
    # sign = 1 if x >= 0 else -1
    sign = np.sign(x);
    x    = np.abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    return sign*y

# def erf(x):
#     xr = x.reshape(x.size)
#     y  = 1.0 * xr
#     for i in range(x.size):
#         print(xr[i])
#         print(erf_scalar(xr[i]))
#         y[i] = erf_scalar(xr[i])
#     print(y)
#     return y.reshape(x.shape)

class NN:
    def __init__(self,dim, layer_sizes, activations):
        self.num_layers  = np.copy(len(layer_sizes))
        self.layer_sizes = np.copy(layer_sizes)
        self.activation  = activations
        self.dim         = dim
    def num_param(self):
        xs = [self.dim];
        np = 0;
        for ls in self.layer_sizes:
            xs.append(ls)
        for i in range(self.num_layers):
            np += (1+xs[i]) * xs[i+1]
        return np

    def predict(self, w, x):
        dim, num_data = x.shape
        out           = x;
        bias          = np.ones((1, num_data));
        prev_size     = dim
        start_idx     = 0;
        for i in range(self.num_layers):
            layer_size   = self.layer_sizes[i]
            num_w_layer  = (prev_size+1) * layer_size;
            w_layer      = np.reshape(w[start_idx:start_idx+num_w_layer], (prev_size+1, layer_size))
            out          = self.activation[i](np.dot(w_layer.T, np.concatenate((bias, out))))
            prev_size    = layer_size
            start_idx   += num_w_layer
        return out


def chol_solve(L, y):
    """
    K = L.dot(L.T)
    return inv(K) * y
    """
    v = np.linalg.solve(L, y)
    return np.linalg.solve(L.T, v)


def chol_inv(L):
    return chol_solve(L, np.eye(L.shape[0]))

class DSK_GP:
    def __init__(self, train_x, train_y, layer_sizes, activations, bfgs_iter=500, l1=0, l2=0, debug = False):
        self.train_x   = np.copy(train_x)
        self.train_y   = np.copy(train_y)
        self.dim       = self.train_x.shape[0]
        self.num_train = self.train_x.shape[1]
        self.nn        = NN(self.dim, layer_sizes, activations)
        self.num_param = 2 + self.dim + self.nn.num_param() # noise + variance + lengthscales + NN weights
        self.m         = layer_sizes[-1];
        self.loss      = np.inf
        self.bfgs_iter = bfgs_iter;
        self.debug     = debug
        self.l1        = l1; # TODO: only regularize weight, do not regularize bias
        self.l2        = l2;
        self.train_y.reshape(1, train_y.size)

    def scale_x(self, train_x, log_lscales):
        lscales = np.exp(log_lscales).repeat(train_x.shape[1], axis=0).reshape(train_x.shape);
        return train_x / lscales

    def log_likelihood(self, theta):
        # TODO: verification of this log_likelihood
        log_sn       = theta[0]
        log_sp       = theta[1]
        log_lscales  = theta[2:2+self.dim];
        w            = theta[2+self.dim:]
        scaled_x     = self.scale_x(self.train_x, log_lscales)
        sn2          = np.exp(2 * log_sn)
        sp           = np.exp(1 * log_sp);
        sp2          = np.exp(2 * log_sp);
        # Phi          = sp * self.nn.predict(w, self.train_x) / np.sqrt(self.m)
        Phi          = sp * self.nn.predict(w, scaled_x) / np.sqrt(self.m)
        m, num_train = Phi.shape
        A            = sn2 * np.eye(m) + np.dot(Phi, Phi.T)
        LA           = np.linalg.cholesky(A)

        # data fit
        data_fit_1 = np.dot(self.train_y, self.train_y.T)
        data_fit_2 = np.dot(Phi, self.train_y.T)
        data_fit_2 = chol_solve(LA, data_fit_2)
        data_fit_2 = np.dot(Phi.T, data_fit_2)
        data_fit_2 = np.dot(self.train_y, data_fit_2)
        data_fit   = (data_fit_1 - data_fit_2) / sn2;


        # model complexity
        # TODO: sum up the diagonal of LA to calculate logDetA
        s, logDetA       = np.linalg.slogdet(A)
        model_complexity = (num_train - m) * (2 * log_sn) + logDetA;

        neg_likelihood   = 0.5 * (data_fit + model_complexity + num_train * np.log(2 * np.pi))
        return neg_likelihood

    def fit(self, theta, optimize=True):
        theta0     = theta.copy()
        self.loss  = np.inf
        self.theta = theta0;
        def loss(w):
            nlz    = self.log_likelihood(w);
            l1_reg = self.l1 * np.abs(w).sum();
            l2_reg = self.l2 * np.dot(w.reshape(1, w.size), w.reshape(w.size, 1))
            loss   = nlz + l1_reg + l2_reg
            if loss < self.loss:
                self.loss  = loss
                self.theta = w
            return loss
        gloss      = grad(loss)
        try:
            if optimize:
                fmin_l_bfgs_b(loss, theta0, gloss, maxiter = self.bfgs_iter, m=100, iprint=10)
        except:
            print("Exception caught, L-BFGS early stopping...")
            print(sys.exc_info())

        print("Optimized")

        # pre-computation
        log_sn = self.theta[0]
        log_sp = self.theta[1]
        log_lscales = self.theta[2:2+self.dim]
        w      = self.theta[2+self.dim:]
        sn2    = np.exp(2 * log_sn)
        sp     = np.exp(log_sp);
        sp2    = np.exp(2*log_sp);
        Phi    = self.nn.predict(w, self.scale_x(self.train_x, log_lscales))
        m      = self.m
        A      = (sn2 * m / sp2) * np.eye(m) + np.dot(Phi, Phi.T)
        LA     = np.linalg.cholesky(A)

        self.LA     = LA.copy()
        self.alpha  = chol_solve(LA, np.dot(Phi, self.train_y.T))
        # A      = sn2 * np.eye(m) + np.dot(Phi, Phi.T)
        # LA     = np.linalg.cholesky(A)

        # alpha      = self.train_y.copy()
        # alpha      = Phi.dot(alpha.T)
        # alpha      = chol_solve(LA, alpha)
        # alpha      = Phi.T.dot(alpha)
        # alpha      = self.train_y.T - alpha;
        # alpha      = alpha / sn2;
        # alpha      = Phi.dot(alpha)
        # self.alpha = alpha

        # Qmm    = Phi.dot(Phi.T)
        # self.B = (Qmm - Qmm.dot(chol_solve(LA, Qmm))) / sn2
        # if self.debug:
        #     np.savetxt('B', self.B)

    def predict(self, test_x):
        log_sn      = self.theta[0]
        log_sp      = self.theta[1]
        log_lscales = self.theta[2:2+self.dim]
        w           = self.theta[2+self.dim:]
        sn          = np.exp(log_sn)
        sn2         = np.exp(2*log_sn)
        sp          = np.exp(log_sp)
        sp2         = np.exp(2*log_sp)
        Phi_test    = self.nn.predict(w, self.scale_x(test_x, log_lscales))
        py          = Phi_test.T.dot(self.alpha)
        ps2         = sn2 + sn2 * np.diagonal(Phi_test.T.dot(chol_solve(self.LA, Phi_test)));
        # ps2      = np.diagonal(np.exp(2 * log_sn) + Phi_test.T.dot(Phi_test) - Phi_test.T.dot(self.B.dot(Phi_test)))
        # if self.debug:
        #     np.savetxt('sf2', np.diagonal(Phi_test.T.dot(Phi_test)))
        return py, ps2


# def f(x):
#     xsum = x.sum(axis=0);
#     # return np.sign(xsum)
#     return 0.05 * xsum**2 + np.sin(xsum);

# dim       = 1
# num_train = 600
# num_test  = 1000
# sn        = 1e-1
# train_x   = 5 * np.random.randn(dim, num_train)
# train_y   = f(train_x) + sn * np.random.randn(1, num_train)
# test_x    = np.linspace(-30, 60, num_test).reshape(dim, num_test);
# test_y    = f(test_x).reshape(1, num_test);
# gp        = DSK_GP(train_x, train_y, [50, 50], [tanh, tanh])


# theta   = np.random.randn(gp.num_param)
# gloss   = grad(gp.log_likelihood)
# g_theta = gloss(theta)

# gp.fit(theta)
# py, ps2 = gp.predict(test_x)

# plt.plot(test_x[0], py.T[0], 'b')
# plt.plot(test_x[0], py.T[0] + np.sqrt(ps2.T[0]), 'g')
# plt.plot(test_x[0], py.T[0] - np.sqrt(ps2.T[0]), 'g')
# plt.plot(train_x, train_y, 'r+')
# plt.show()

# np.savetxt('train_x', train_x)
# np.savetxt('train_y', train_y)
# np.savetxt('test_x', test_x)
# np.savetxt('test_y', test_y)
# np.savetxt('pred_y', py)
# np.savetxt('pred_s', np.sqrt(ps2))
# np.savetxt('theta', gp.theta)
