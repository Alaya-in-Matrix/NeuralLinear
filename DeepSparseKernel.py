import autograd.numpy as np
from autograd import grad
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
import math
import sys
from scipy.optimize import fmin_cg, fmin_l_bfgs_b, fmin_ncg

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
            out          = self.activation[i](np.dot(w_layer.T, np.concatenate((out, bias))))
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
        self.mean      = np.mean(train_y)
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
        self.train_y_zero = self.train_y - self.mean;

    def scale_x(self, train_x, log_lscales):
        lscales = np.exp(log_lscales).repeat(train_x.shape[1], axis=0).reshape(train_x.shape);
        return train_x / lscales

    def calc_Phi(self, w, x):
        Phi = self.nn.predict(w, x);
        return Phi

    def log_likelihood(self, theta):
        # TODO: verification of this log_likelihood
        log_sn      = theta[0]
        log_sp      = theta[1]
        log_lscales = theta[2:2+self.dim];
        w           = theta[2+self.dim:]
        scaled_x    = self.scale_x(self.train_x, log_lscales)
        sn2         = np.exp(2 * log_sn)
        sp          = np.exp(1 * log_sp);
        sp2         = np.exp(2 * log_sp);

        neg_likelihood = np.inf
        Phi            = self.calc_Phi(w, scaled_x);
        m, num_train   = Phi.shape
        A              = np.dot(Phi, Phi.T) + (sn2 * m / sp2) * np.eye(m);
        LA             = np.linalg.cholesky(A)

        Phi_y = np.dot(Phi, self.train_y_zero.T)
        data_fit = (np.dot(self.train_y_zero, self.train_y_zero.T) - np.dot(Phi_y.T, chol_solve(LA, Phi_y))) / sn2
        logDetA = 0
        for i in range(m):
            logDetA += 2 * np.log(LA[i][i])
        neg_likelihood = 0.5 * (data_fit + logDetA - m * np.log(m * sn2 / sp2) + num_train * np.log(2 * np.pi * sn2))
        if(np.isnan(neg_likelihood)):
            neg_likelihood = np.inf
        
        l1_reg         = self.l1 * np.abs(w).sum();
        l2_reg         = self.l2 * np.dot(w.reshape(1, w.size), w.reshape(w.size, 1))
        neg_likelihood = neg_likelihood + l1_reg + l2_reg

        # refresh current best
        if neg_likelihood < self.loss:
            self.loss  = loss.copy()
            self.theta = theta.copy()
            self.LA    = LA.copy()
            self.A     = A.copy()

        return neg_likelihood

    def fit(self, theta, optimize=True):
        theta0     = theta.copy()
        self.loss  = np.inf
        self.theta = theta0;
        def loss(w):
            nlz = self.log_likelihood(w);
            return nlz
        gloss      = grad(loss)
        try:
            if optimize:
                fmin_l_bfgs_b(loss, theta0, gloss, maxiter = self.bfgs_iter, m = 100, iprint=1)
        except:
            print("Exception caught, L-BFGS early stopping...")
            print(sys.exc_info())

        print("Optimized loss is %g" % self.loss)
        if(np.isinf(self.loss) or np.isnan(self.loss)):
            print("Fail to build GP model")
            sys.exit(1)

        # pre-computation
        log_sn      = self.theta[0]
        log_sp      = self.theta[1]
        log_lscales = self.theta[2:2+self.dim]
        w           = self.theta[2+self.dim:]
        sn2         = np.exp(2 * log_sn)
        sp          = np.exp(log_sp);
        sp2         = np.exp(2*log_sp);
        Phi         = self.calc_Phi(w, self.scale_x(self.train_x, log_lscales))
        m           = self.m
        self.alpha  = chol_solve(self.LA, np.dot(Phi, self.train_y_zero.T))

    def predict(self, test_x):
        log_sn      = self.theta[0]
        log_sp      = self.theta[1]
        log_lscales = self.theta[2:2+self.dim]
        w           = self.theta[2+self.dim:]
        sn          = np.exp(log_sn)
        sn2         = np.exp(2*log_sn)
        sp          = np.exp(log_sp)
        sp2         = np.exp(2*log_sp)
        Phi_test    = self.calc_Phi(w, self.scale_x(test_x, log_lscales))
        py          = self.mean + Phi_test.T.dot(self.alpha)
        ps2         = sn2 + sn2 * np.diagonal(Phi_test.T.dot(chol_solve(self.LA, Phi_test)));
        return py, ps2
