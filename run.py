from DeepSparseKernel import *
import matplotlib.pyplot as plt

train_x = np.loadtxt('train_x').T
train_y = np.loadtxt('train_y')
train_y = train_y.reshape(1, train_y.size)

test_x = np.loadtxt('test_x').T
test_y = np.loadtxt('test_y')
test_y = test_y.reshape(1, test_y.size)

layer_sizes = [50, 50]
activations = [relu, relu]
scale       = 0.5

gp       = DSK_GP(train_x, train_y, layer_sizes, activations);
theta    = scale * np.random.randn(gp.num_param)
theta[0] = np.log(np.std(1e-2 * train_y))
gp.fit(theta)
py, ps2 = gp.predict(test_x)

np.savetxt('pred_y', py)
np.savetxt('pred_s', np.sqrt(np.maximum(ps2, 0.0)))
np.savetxt('theta', gp.theta)

plt.plot(test_y.reshape(test_y.size), py.reshape(py.size), 'r.')
plt.show()
