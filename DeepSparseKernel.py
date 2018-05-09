import autograd.numpy as np
from autograd import grad
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
import math
import sys
from scipy.optimize import fmin_cg, fmin_l_bfgs_b, fmin_ncg
import traceback

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
    def __init__(self, layer_sizes, activations):
        self.num_layers  = np.copy(len(layer_sizes))
        self.layer_sizes = np.copy(layer_sizes)
        self.activation  = activations
    def num_param(self, xdim):
        xs = [xdim];
        np = 0;
        for ls in self.layer_sizes:
            xs.append(ls)
        for i in range(self.num_layers):
            np += (1+xs[i]) * xs[i+1]
        return np

    def w_nobias(self, w, dim):
        """
        return weights without bias, it can be used for the L1/L2 regularizaton
        """
        prev_size = dim
        start_idx = 0;
        wnb       = np.array([])
        for i in range(self.num_layers):
            layer_size   = self.layer_sizes[i]
            num_w_layer  = (prev_size+1) * layer_size;
            w_layer      = np.reshape(w[start_idx:start_idx+num_w_layer], (prev_size+1, layer_size))[:prev_size, :];
            wnb          = np.concatenate((wnb, w_layer.reshape(w_layer.size)));
            prev_size    = layer_size
            start_idx   += num_w_layer
        return wnb

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


def scale_x(xs, log_lscales):
    lscales = np.exp(log_lscales).repeat(xs.shape[1], axis=0).reshape(xs.shape);
    return xs / lscales

class DSK_GP:
    def __init__(self, train_x, train_y, layer_sizes, activations, bfgs_iter=500, l1=0, l2=0, debug = False):
        self.train_x   = np.copy(train_x)
        self.train_y   = np.copy(train_y)
        self.mean      = np.mean(train_y)
        self.dim       = self.train_x.shape[0]
        self.num_train = self.train_x.shape[1]
        self.nn        = NN(layer_sizes, activations)
        self.num_param = 2 + self.dim + self.nn.num_param(self.dim) # noise + variance + lengthscales + NN weights
        self.m         = layer_sizes[-1];
        self.loss      = np.inf
        self.bfgs_iter = bfgs_iter;
        self.debug     = debug
        self.l1        = l1; # TODO: only regularize weight, do not regularize bias
        self.l2        = l2;
        self.train_y.reshape(1, train_y.size)
        self.train_y_zero = self.train_y - self.mean;

    def rand_theta(self, scale=0.1):
        """ 
        Generate an initial theta, the weights of NN are randomly initialized
        """
        theta = scale * np.random.randn(self.num_param)
        theta[0] = np.log(np.std(self.train_y) / 2)
        theta[1] = np.log(np.std(self.train_y))
        for i in range(self.dim):
            theta[2 *  + i] = np.maximum(-100, np.log(0.5 * (self.train_x[i, :].max() - self.train_x[i, :].min())))
        return theta

    def calc_Phi(self, w, x):
        Phi = self.nn.predict(w, x);
        return Phi

    def log_likelihood(self, theta):
        # TODO: verification of this log_likelihood
        log_sn      = theta[0]
        log_sp      = theta[1]
        log_lscales = theta[2:2+self.dim];
        w           = theta[2+self.dim:]
        scaled_x    = scale_x(self.train_x, log_lscales)
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
        
        w_nobias       = self.nn.w_nobias(w, self.dim);
        l1_reg         = self.l1 * np.abs(w_nobias).sum()
        l2_reg         = self.l2 * np.dot(w_nobias, w_nobias)
        neg_likelihood = neg_likelihood + l1_reg + l2_reg

        # refresh current best
        if neg_likelihood < self.loss:
            self.loss  = neg_likelihood
            self.theta = np.copy(theta)
            self.LA    = LA.copy()
            self.A     = A.copy()

        return neg_likelihood

    def fit(self, theta):
        theta0     = theta.copy()
        self.loss  = np.inf
        self.theta = theta0;
        def loss(w):
            nlz = self.log_likelihood(w);
            return nlz
        gloss      = grad(loss)
        try:
            fmin_l_bfgs_b(loss, theta0, gloss, maxiter = self.bfgs_iter, m = 100, iprint=1)
        except np.linalg.LinAlgError:
            print("Increase noise term and re-optimization")
            theta0     = np.copy(self.theta);
            theta0[0] += np.log(10);
            try:
                fmin_l_bfgs_b(loss, theta0, gloss, maxiter = self.bfgs_iter, m = 10, iprint=1)
            except:
                print("Exception caught, L-BFGS early stopping...")
                if self.debug:
                    print(traceback.format_exc())
        except:
            print("Exception caught, L-BFGS early stopping...")
            if self.debug:
                print(traceback.format_exc())

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
        Phi         = self.calc_Phi(w, scale_x(self.train_x, log_lscales))
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
        Phi_test    = self.calc_Phi(w, scale_x(test_x, log_lscales))
        py          = self.mean + Phi_test.T.dot(self.alpha)
        ps2         = sn2 + sn2 * np.diagonal(Phi_test.T.dot(chol_solve(self.LA, Phi_test)));
        return py, ps2


class MODSK:
    def __init__(self, train_x, train_y, shared_nn, non_shared_nns, max_iter = 100, l1 = 0, l2 = 0, debug=False): 
        self.train_x        = np.copy(train_x)
        self.train_y        = np.copy(train_y)
        self.dim            = self.train_x.shape[0]
        self.num_train      = self.train_x.shape[1]
        self.num_obj        = self.train_y.shape[1]
        self.means          = np.mean(self.train_y, axis=0)
        self.stds           = np.std(self.train_y, axis=0)
        self.train_y        = (self.train_y - self.means) / self.stds # standardize output
        self.debug          = debug
        self.max_iter       = max_iter # max iter for the L-BFGS optimization
        self.l1             = l1
        self.l2             = l2
        self.shared_nn      = shared_nn
        self.non_shared_nns = non_shared_nns
        self.num_param      = self.calc_num_params()
        if(train_x.ndim != 2 or train_y.ndim != 2):
            print("train_x.ndim != 2 or train_y.ndim != 2")
            sys.exit(1)
        if(train_x.shape[1] != train_y.shape[0]):
            print("train_x.shape[1] != train_y.shape[0]")
            sys.exit(1)
        if(len(non_shared_nns) != self.num_obj):
            print("len(non_shared_nns) != self.num_obj")
            sys.exit(1)

    def calc_num_params(self):
        """ 
        parameters:
            1. length scales: dim
            2. noise: num_obj
            3. self covariance: num_obj
        """
        num_param = self.dim + 2 * self.num_obj + self.shared_nn.num_param(self.dim)
        size_last_layer_shared = self.shared_nn.layer_sizes[-1]
        for i in range(self.num_obj):
            num_param += self.non_shared_nns[i].num_param(size_last_layer_shared)
        return num_param

    def w_nobias(self, ws):
        w_shared     = ws[:self.shared_nn.num_param(self.dim)]
        w_non_shared = ws[self.shared_nn.num_param(self.dim):]
        m_shared     = self.shared_nn.layer_sizes[-1]
        wnb          = self.shared_nn.w_nobias(w_shared, self.dim)
        start_idx    = 0
        for nn in self.non_shared_nns:
            w_tmp = w_non_shared[start_idx: start_idx + nn.num_param(m_shared)]
            wnb   = np.concatenate((wnb, nn.w_nobias(w_tmp, m_shared)))
            start_idx = start_idx + w_tmp.size
            if self.debug:
                assert(w_tmp.size == nn.num_param(m_shared))
        return wnb
    def calc_Phi(self, ws, x):
        w_shared     = ws[:self.shared_nn.num_param(self.dim)]
        w_non_shared = ws[self.shared_nn.num_param(self.dim):]
        Phi_shared   = self.shared_nn.predict(w_shared, x)
        m_shared     = Phi_shared.shape[0]
        Phis         = []
        start_idx    = 0
        for nn in self.non_shared_nns:
            w_tmp     = w_non_shared[start_idx: start_idx + nn.num_param(m_shared)]
            Phi_tmp   = nn.predict(w_tmp, Phi_shared)
            start_idx = start_idx + w_tmp.size
            Phis     += [Phi_tmp]
            if self.debug:
                assert(w_tmp.size == nn.num_param(Phi_shared.shape[0]))
        return Phis

    def split_theta(self, theta):
        if self.debug:
            assert(theta.size == self.num_param)
        num_obj     = self.num_obj
        log_sns     = theta[:num_obj];
        log_sps     = theta[num_obj:2*num_obj];
        log_lscales = theta[2*num_obj:2*num_obj+self.dim]
        ws          = theta[2*num_obj+self.dim:]
        return (log_sns, log_sps, log_lscales, ws)

    def rand_theta(self, scale=1):
        """ 
        Generate an initial theta, the weights of NN are randomly initialized
        """
        theta = scale * np.random.randn(self.num_param)
        # noises and self covariances
        for i in range(self.num_obj):
            theta[i]                = np.log(np.std(self.train_y[:, i]) / 2)
            theta[self.num_obj + i] = np.log(np.std(self.train_y[:, i]))
        # lengthscales
        for i in range(self.dim):
            theta[2 * self.num_obj + i] = np.maximum(-100, np.log(0.5 * (self.train_x[i, :].max() - self.train_x[i, :].min())))
        return theta
    
    def loss(self, theta):
        """
        return a list of losses
        """
        if self.debug:
            assert(theta.size == self.num_param)
        log_sns, log_sps, log_lscales, ws = self.split_theta(theta)
        Phis                              = self.calc_Phi(ws, scale_x(self.train_x, log_lscales))
        losses                            = []
        for i in range(self.num_obj):
            losses += [self.log_likelihood(log_sns[i], log_sps[i], Phis[i], self.train_y[:, i].reshape(self.num_train, 1))]
        return losses

    def log_likelihood(self, log_sn, log_sp, Phi, train_y):
        sn2         = np.exp(2 * log_sn)
        sp          = np.exp(1 * log_sp);
        sp2         = np.exp(2 * log_sp);

        neg_likelihood = np.inf
        m, num_train   = Phi.shape
        A              = np.dot(Phi, Phi.T) + (sn2 * m / sp2) * np.eye(m);
        LA             = np.linalg.cholesky(A)

        Phi_y = np.dot(Phi, train_y)
        data_fit = (np.dot(train_y.T, train_y) - np.dot(Phi_y.T, chol_solve(LA, Phi_y))) / sn2
        logDetA = 0
        for i in range(m):
            logDetA += 2 * np.log(LA[i][i])
        neg_likelihood = 0.5 * (data_fit + logDetA - m * np.log(m * sn2 / sp2) + num_train * np.log(2 * np.pi * sn2))
        if(np.isnan(neg_likelihood)):
            neg_likelihood = np.inf
        
        return neg_likelihood

    def fit(self, theta):
        theta0         = theta.copy()
        self.best_loss = np.inf
        def lossfit(theta):
            loss   = sum(self.loss(theta))
            w      = theta[2*self.num_obj+self.dim:]
            wnb    = self.w_nobias(w)
            l1_reg = self.l1 * np.abs(wnb).sum();
            l2_reg = self.l2 * np.dot(wnb, wnb)
            loss   = loss + l1_reg + l2_reg
            if loss < self.best_loss:
                self.best_loss = loss
                self.theta     = theta.copy()
            return loss
        gloss = grad(lossfit)
        try:
            fmin_l_bfgs_b(lossfit, theta0, gloss, maxiter = self.max_iter, m = 100, iprint=1)
        except np.linalg.LinAlgError:
            print("Increase noise term and re-optimization")
            theta0 = np.copy(self.theta)
            for i in range(self.num_obj):
                theta0[i] = theta0[i] + np.log(10)
            try:
                fmin_l_bfgs_b(lossfit, theta0, gloss, maxiter = self.max_iter, m = 10, iprint=1)
            except:
                print("Exception caught, L-BFGS early stopping...")
                if self.debug:
                    print(traceback.format_exc())
        except:
            print("Exception caught, L-BFGS early stopping...")
            if self.debug:
                print(traceback.format_exc())
        
        print("Optimized")
        log_sns, log_sps, log_lscales, ws = self.split_theta(self.theta)
        scaled_x                          = scale_x(self.train_x, log_lscales)
        Phis                              = self.calc_Phi(ws, scaled_x)
        self.Phis                         = Phis
        self.LAs                          = []
        self.alphas                       = []
        for i in range(self.num_obj):
            Phi          = Phis[i]
            sn2          = np.exp(2 * log_sns[i])
            sp2          = np.exp(2 * log_sps[i])
            m            = Phi.shape[0]
            A            = np.dot(Phi, Phi.T) + (sn2 * m / sp2) * np.eye(m);
            LA           = np.linalg.cholesky(A)
            self.LAs    += [LA]
            self.alphas += [chol_solve(LA, np.dot(Phi, self.train_y[:, i]))]

    def predict(self, x):
        num_test = x.shape[1];
        py       = np.zeros((num_test, self.num_obj))
        ps2      = np.zeros((num_test, self.num_obj))

        log_sns, log_sps, log_lscales, ws = self.split_theta(self.theta)
        scaled_x  = scale_x(x, log_lscales)
        Phis_test = self.calc_Phi(ws, scaled_x)

        for i in range(self.num_obj):
            Phi_test = Phis_test[i]
            sn2      = np.exp(2 * log_sns[i])
            py[:, i] = np.dot(Phi_test.T, self.alphas[i]);
            for j in range(num_test):
                ps2[j, i] = sn2 + sn2 * np.dot(Phi_test[:, j].T, chol_solve(self.LAs[i], Phi_test[:, j]))
        py  = (py * self.stds) + self.means;
        ps2 = ps2 * (self.stds**2)
        return py, ps2

    def mix_predict(self, K, x, scale):
        # TODO: this version can not be paralleled, as different threads would share the same self.theta
        pys    = []
        ps2s   = []
        losses = []
        for i in range(K):
            theta   = self.rand_theta(scale=scale)
            self.fit(theta)
            py_i, ps2_i  = self.predict(x)
            pys         += [py_i]
            ps2s        += [ps2_i]
            losses      += [self.best_loss[0][0]]
        py  = np.zeros((x.shape[1], self.num_obj))
        ps2 = np.zeros((x.shape[1], self.num_obj))
        for i in range(K):
            py  += pys[i] / K;
            ps2 += (ps2s[i] + pys[i]**2) / K
        ps2 -= py**2
        print("losses:")
        print(losses)
        return py, ps2

# TODO: # https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e
