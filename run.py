from duck import *
from DeepSparseKernel import *

w      = np.random.randn((dim+1) * m);
sn     = 1e-2
log_sn = np.log(sn)
theta  = np.concatenate((np.array([log_sn]), w))

duck_loss = loss(w)
print(duck_loss)


gp      = DSK_GP(train_x, train_y, [m], [tanh])
gp_loss = gp.log_likelihood(theta)
print(gp_loss)
