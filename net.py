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
flags.DEFINE_boolean("use_rotation", True, "if True, load rotated data")
flags.DEFINE_integer("epochs", 10, "number of epochs")
flags.DEFINE_boolean("use_model", True, "if True, use model of trained")

class Linear(object):
    def __init__(self, in_dim, out_dim, name):
        with tf.variable_scope('linear'):
            self.W = tf.get_variable("{}w".format(name), shape=[in_dim, out_dim],
                    initializer=tf.constant_initializer(np.random.uniform(low=-np.sqrt(6./(in_dim+out_dim)), high=np.sqrt(6./(in_dim+out_dim)), size=[in_dim, out_dim])))
            self.b = tf.get_variable("{}b".format(name), shape=[out_dim],
                    initializer=tf.constant_initializer(np.zeros(shape=[out_dim])))

    def __call__(self, x):
        return tf.matmul(x, self.W)+self.b

class BatchNorm(object):
    def __init__(self, shape, name, epsilon=tf.constant(1e-5)):
        self.epsilon = epsilon
        if isinstance(shape, int):
            shape = [shape]
        self.gamma = tf.get_variable("{}gamma".format(name), shape=shape, initializer=tf.constant_initializer(np.ones(shape=shape)))
        self.beta = tf.get_variable("{}beta".format(name), shape=shape, initializer=tf.constant_initializer(np.zeros(shape=shape)))

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

            self.KL = -1/2*tf.reduce_mean(tf.reduce_sum(1+tf.log(tf.clip_by_value(self.var, 1e-10, 1.0))-self.mean**2-self.var, reduction_indices=[1]))
            epsilon = tf.random_normal(shape=[z_dim])
            self.z = self.mean+self.var*epsilon
            self.pi = self.p_x_yz(self.y, self.z)
            tf.get_variable_scope().reuse_variables()
            self.pi_out = self.p_x_yz(self.y, self.z_in)
            self.log_likelihood = tf.reduce_mean(self.log_p_x_yz(self.pi, self.x)+self.log_p_y(self.y))
            self.lower_bound = -self.KL+self.log_likelihood
            self.loss = -self.lower_bound
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def training(self, x, y, sess):
        sess.run(self.train_op, feed_dict={self.x: x, self.y: y})

    def loss_op(self, x, y, sess):
        return sess.run(self.loss, feed_dict={self.x: x, self.y: y})

    def reconstruct(self, x, y, sess):
        if x.ndim == 1 or y.ndim == 1:
            x = x.reshape(1, self.image_dim)
            y = y.reshape(1, self.cat_dim)
        return sess.run(self.pi, feed_dict={self.x: x, self.y: y})

    def struct_img(self, y, z, sess):
        return sess.run(self.pi_out, feed_dict={self.y: y, self.z_in: z})

    def log_p_y(self, y):
        return tf.log(tf.fill([1], 1/self.cat_dim))

    def log_p_x_yz(self, pi, x):
        return tf.reduce_sum(x*tf.log(tf.clip_by_value(pi, 1e-10, 1.0))+(1-x)*tf.log(tf.clip_by_value(1-pi, 1e-10, 1.0)), reduction_indices=[1])

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
                var = Activation(tf.nn.sigmoid)(var)

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
                #pi = BatchNorm(self.image_dim, 'bn1')(pi)
                pi = Activation(function=tf.nn.sigmoid)(pi)

                return pi

def rotation(X, y):
    img    = X[0].reshape(-1, 28)
    center = (img.shape[0]*0.5, img.shape[1]*0.5)
    size   = (img.shape[0], img.shape[1])
    scale  = 1.0

    y = np.concatenate((y, np.zeros(shape=(y.shape[0], 1))), axis=1)
    length = X.shape[0]

    for i, x in enumerate(X):
        print i
        x = x.reshape(-1, 28)
        angle = np.random.randint(0, 180)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        x_rot = cv2.warpAffine(x, rotation_matrix, size)
        x_rot = x_rot.reshape(784)
        X[i] = x_rot
        y[i, 10] = angle/180.0

    return X, y

def imshow10(x1, x2, fig):
    if len(x1) > 10:
        x1 = x1[0:10]
        x1 = x1[0:10]
    for i, (xx1, xx2) in enumerate(zip(x1, x2)):
        ax = fig.add_subplot(2, 10, 2*i+1)
        ax.imshow(xx1.reshape(28, 28))
        ax = fig.add_subplot(2, 10, 2*i+2)
        ax.imshow(xx2.reshape(28, 28))


if __name__ == '__main__':
    if FLAGS.use_rotation:
        mnist = np.load('rotation.npz')
        mnist_X = mnist['x']
        mnist_y = mnist['y']
        print 'load mnist'
    else:
        pass
        mnist = fetch_mldata('MNIST original')
        mnist_X, mnist_y = mnist.data, mnist.target.astype(np.int32)
        mnist_X = mnist_X/255.0
        mnist_y = np.eye(np.max(mnist_y)+1)[mnist_y]
        mnist_X, mnist_y = rotation(mnist_X, mnist_y)
        print 'finish rotation'
        np.savez('rotation.npz', x=mnist_X, y=mnist_y)


    train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y, test_size=0.2)

    vae = VAE()

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    if FLAGS.use_model:
        saver.restore(sess, 'model.ckpt')
    else:
        batch_size = 100
        for epoch in range(FLAGS.epochs):
            train_X, train_y = shuffle(train_X, train_y)
            for i in range(train_X.shape[0]//batch_size):
                vae.training(train_X[i*batch_size:(i+1)*batch_size], train_y[i*batch_size:(i+1)*batch_size], sess)
            print "epoch: {}, loss: {}".format(epoch, vae.loss_op(test_X, test_y, sess))
        saver.save(sess, 'model.ckpt')

    reconstruct_image = vae.reconstruct(test_X, test_y, sess)
    fig = plt.figure(figsize=(10, 10))
    imshow10(test_X, reconstruct_image, fig)

    plt.savefig('reconstruct.png')

