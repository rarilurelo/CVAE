import numpy as np

mnist = np.load('augmented.npz')
mnist_X = mnist['x']
mnist_y = mnist['y']

print np.isnan(mnist_X).any()
