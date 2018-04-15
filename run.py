import DeepSparseKernel as dsk
from DeepSparseKernel import np
import matplotlib.pyplot as plt

train_x = np.loadtxt('train_x').T
train_y = np.loadtxt('train_y')
train_y = train_y.reshape(1, train_y.size)

test_x = np.loadtxt('test_x').T
test_y = np.loadtxt('test_y')
test_y = test_y.reshape(1, test_y.size)

layer_sizes = [300]
activations = [dsk.tanh, dsk.relu]
scale       = 0.1

gp       = dsk.DSK_GP(train_x, train_y, layer_sizes, activations, bfgs_iter=100);
theta    = scale * np.random.randn(gp.num_param)
theta[0] = np.log(np.std(1e-2 * train_y))
gp.fit(theta)
py, ps2 = gp.predict(test_x)

Phi_train = gp.nn.predict(gp.theta[1:], train_x);
Phi_test  = gp.nn.predict(gp.theta[1:], test_x);

np.savetxt('pred_y', py)
np.savetxt('pred_s', np.sqrt(np.maximum(ps2, 0.0)))
np.savetxt('theta', gp.theta)
np.savetxt('Phi_train', Phi_train)
np.savetxt('Phi_test', Phi_test)

plt.plot(test_y.reshape(test_y.size), py.reshape(py.size), 'r.')
plt.show()

gp.debug = True
print(gp.log_likelihood(gp.theta))
