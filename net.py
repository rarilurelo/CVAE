from __future__ import division
import numpy as np
import tensorflow as tf
import cv2
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean("use_augmentation", False, "if True load augmented data")
flags.DEFINE_integer("epochs", 10, "number of epochs")

class Linear(object):
    def __init__(self, in_dim, out_dim, name):
        with tf.variable_scope('linear'):
            self.W = tf.get_variable(name, shape=[in_dim, out_dim],
                    initializer=tf.constant_initializer(np.random.uniform(low=-np.sqrt(6./(in_dim+out_dim)), high=np.sqrt(6./(in_dim+out_dim)))))

    def __call__(self, x):
        return tf.matmul(x, self.W)

class BatchNorm(object):
    def __init__(self, shape, name, epsilon=tf.constant(1e-5)):
        self.epsilon = epsilon
        self.gamma = tf.get_variable("{}gamma".format(name), shape=[shape], initializer=tf.constant_initializer(np.ones(shape=[shape])))
        self.beta = tf.get_variable("{}beta".format(name), shape=[shape], initializer=tf.constant_initializer(np.zeros(shape=[shape])))

    def __call__(self, x):
        if x.get_shape().ndims == 2:
            mean, var = tf.nn.moments(x, axes=(0,), keep_dims=True)
        elif x.get_shape().ndims == 4:
            mean, var = tf.nn.moments(x, axes=(0, 2, 3), keep_dims=True)
        normalized_x = (x-mean)/tf.sqrt(var)
        return self.gamma*normalized_x+self.beta

class Activation(object):
    def __init__(self, function=tf.nn.relu):
        self.function = function

    def __call__(self, x):
        return self.function(x)

class VAE(object):
    def __init__(self, image_dim=784, cat_dim=11, z_dim=50, hid_dim=200):
        self.image_dim = image_dim
        self.cat_dim = cat_dim
        self.z_dim = z_dim
        self.hid_dim = hid_dim

        self.x = tf.placeholder(dtype=tf.float32, shape=(None, image_dim), name='x')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, cat_dim), name='y')
        self.z_in = tf.placeholder(dtype=tf.float32, shape=(None, z_dim), name='z_in')
        with tf.variable_scope('vae'):
            self.mean, self.var = self.q_z_xy(self.x, self.y)
            # for debug
            self.log_var = tf.log(self.var)

            self.KL = -1/2*tf.reduce_mean(tf.reduce_sum(1+tf.log(self.var)-self.mean**2-self.var, reduction_indices=[1]))
            epsilon = tf.random_normal(shape=[z_dim])
            self.z = self.mean+self.var*epsilon
            self.pi = self.p_x_yz(self.y, self.z)
            tf.get_variable_scope().reuse_variables()
            self.pi_out = self.p_x_yz(self.y, self.z_in)
            self.log_likelihood = tf.reduce_mean(self.log_p_x_yz(self.pi, self.x)+self.log_p_y(self.y))
            self.lower_bound = -self.KL+self.log_likelihood
            self.loss = -self.lower_bound
            self.train_op = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.1, beta2=0.001).minimize(self.loss)

    def training(self, x, y, sess):
        sess.run(self.train_op, feed_dict={self.x: x, self.y: y})

    def loss_op(self, x, y, sess):
        return sess.run(self.loss, feed_dict={self.x: x, self.y: y})

    def reconstruct(self, x, y, sess):
        if x.ndim == 1 or y.ndim == 1:
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)
        return sess.run(self.pi, feed_dict={self.x: x, self.y: y})

    def struct_img(self, y, z, sess):
        return sess.run(self.pi_out, feed_dict={self.y: y, self.z: z})

    def log_p_y(self, y):
        return tf.fill([1], 1/self.cat_dim)

    def log_p_x_yz(self, pi, x):
        return tf.reduce_sum(x*tf.log(pi)+(1-x)*tf.log(1-pi), reduction_indices=[1])

    def q_z_xy(self, x, y):
        with tf.variable_scope('q_z_xy'):
            with tf.variable_scope('x'):
                x = Linear(self.image_dim, self.hid_dim, 'l1')(x)
                x = BatchNorm(self.hid_dim, 'bn1')(x)
                x = Activation()(x)
                x = Linear(self.hid_dim, self.hid_dim, 'l2')(x)
                x = BatchNorm(self.hid_dim, 'bn2')(x)
                x = Activation()(x)

            with tf.variable_scope('y'):
                y = Linear(self.cat_dim, self.hid_dim, 'l1')(y)
                y = BatchNorm(self.hid_dim, 'bn1')(y)
                y = Activation()(y)
                y = Linear(self.hid_dim, self.hid_dim, 'l2')(y)
                y = BatchNorm(self.hid_dim, 'bn2')(y)
                y = Activation()(y)

            z = x+y

            with tf.variable_scope('z'):
                z = Linear(self.hid_dim, self.hid_dim, 'l1')(z)
                z = BatchNorm(self.hid_dim, 'bn1')(z)
                z = Activation()(z)

            with tf.variable_scope('mean'):
                mean = Linear(self.hid_dim, self.z_dim, 'l1')(z)
                mean = BatchNorm(self.z_dim, 'bn1')(mean)
                mean = Activation()(mean)

            with tf.variable_scope('var'):
                var = Linear(self.hid_dim, self.z_dim, 'l1')(z)
                var = BatchNorm(self.z_dim, 'bn1')(var)
                var = Activation(tf.nn.softplus)(var)

            return mean, var

    def p_x_yz(self, y, z):
        with tf.variable_scope('p_x_yz'):
            with tf.variable_scope('y'):
                y = Linear(self.cat_dim, self.hid_dim, 'l1')(y)
                y = BatchNorm(self.hid_dim, 'bn1')(y)
                y = Activation()(y)
                y = Linear(self.hid_dim, self.hid_dim, 'l2')(y)
                y = BatchNorm(self.hid_dim, 'bn2')(y)
                y = Activation()(y)

            with tf.variable_scope('z'):
                z = Linear(self.z_dim, self.hid_dim, 'l1')(z)
                z = BatchNorm(self.hid_dim, 'bn1')(z)
                z = Activation()(z)
                z = Linear(self.hid_dim, self.hid_dim, 'l2')(z)
                z = BatchNorm(self.hid_dim, 'bn2')(z)
                z = Activation()(z)

            zy = z+y
            with tf.variable_scope('zy'):
                zy = Linear(self.hid_dim, self.hid_dim, 'l1')(zy)
                zy = BatchNorm(self.hid_dim, 'bn1')(zy)
                zy = Activation()(zy)

            with tf.variable_scope('pi'):
                pi = Linear(self.hid_dim, self.image_dim, 'l1')(zy)
                pi = BatchNorm(self.image_dim, 'bn1')(pi)
                pi = Activation(function=tf.nn.sigmoid)(pi)

                return pi

def augmentation(X, y, n=10000):
    img    = X[0].reshape(-1, 28)
    center = (img.shape[0]*0.5, img.shape[1]*0.5)
    size   = (img.shape[0], img.shape[1])
    scale  = 1.0

    y = np.concatenate((y, np.zeros(shape=(y.shape[0], 1))), axis=1)
    length = X.shape[0]

    for _ in range(n):
        ind = np.random.randint(0, X.shape[0])
        x = X[ind]
        new_y = y[ind]
        x = x.reshape(-1, 28)
        angle = np.random.randint(0, 90)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        x_rot = cv2.warpAffine(x, rotation_matrix, size)
        x_rot = x_rot.reshape(1, 784)
        X = np.concatenate((X, x_rot), axis=0)
        new_y[10] = angle/90.0
        new_y = new_y.reshape(1, 11)
        y = np.concatenate((y, new_y), axis=0)

    return X, y




if __name__ == '__main__':
    if FLAGS.use_augmentation:
        mnist = np.load('augmented.npz')
        mnist_X = mnist['x']
        mnist_y = mnist['y']
        print 'load mnist'
    else:
        mnist = fetch_mldata('MNIST original')
        mnist_X, mnist_y = mnist.data, mnist.target.astype(np.int32)
        mnist_X = mnist_X/255.0
        mnist_y = np.eye(np.max(mnist_y)+1)[mnist_y]
        mnist_X, mnist_y = augmentation(mnist_X, mnist_y)
        print 'finish augmentation'
        np.savez('augmented.npz', x=mnist_X, y=mnist_y)


    train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y, test_size=0.2)

    vae = VAE()

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    batch_size = 100
    for epoch in range(FLAGS.epochs):
        train_X, train_y = shuffle(train_X, train_y)
        for i in range(train_X.shape[0]//batch_size):
            vae.training(train_X[i*batch_size:(i+1)*batch_size], train_y[i*batch_size:(i+1)*batch_size], sess)
        # for debug
        feed_dict = {vae.x: test_X, vae.y: test_y}
        log_var = sess.run(vae.log_var, feed_dict)
        kl = sess.run(vae.KL, feed_dict)
        loglike = sess.run(vae.log_likelihood, feed_dict)
        pii = sess.run(vae.pi, feed_dict)
        if np.isnan(kl).any() or np.isinf(kl).any():
            print "kl"
            print kl
            print "log_var"
            print log_var
        if np.isnan(loglike).any() or np.isinf(loglike).any():
            print "loglike"
            print loglike
        if np.isnan(pii).any() or np.isinf(pii).any():
            print "pii"
            print pii
        if np.isnan(log_var).any() or np.isinf(log_var).any():
            print "log_var"
            print log_var

        print "epoch: {}, loss: {}".format(epoch, vae.loss_op(test_X, test_y, sess))

    reconstruct_image = vae.reconstruct(test_X, test_y, sess)
    print test_y[0]
    fig = plt.figure()
    for i in range(5):
        ax = fig.add_subplot(10, 10, 2*i+1)
        ax.imshow(test_X[i].reshape(28, 28))
        ax = fig.add_subplot(10, 10, 2*i+2)
        ax.imshow(reconstruct_image[i].reshape(28, 28))

    plt.show()







