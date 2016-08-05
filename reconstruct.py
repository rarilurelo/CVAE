from __future__ import division
import numpy as np
import tensorflow as tf
import cv2
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from net import VAE

def continuous_imshow10(x, fig):
    if len(x) > 10:
        x = x[0:10]
    for i, xx in enumerate(x):
        ax = fig.add_subplot(1, 10, i+1, xticks=[], yticks=[])
        ax.imshow(xx.reshape(28, 28), 'gray')

if __name__ == '__main__':
    mnist = np.load('rotation.npz')
    mnist_X = mnist['x']
    mnist_y = mnist['y']
    print 'load mnist'


    train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y, test_size=0.2)

    vae = VAE()

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    saver.restore(sess, 'model.ckpt')

    target_X = test_X[0].reshape(1, -1)
    target_y = test_y[0].reshape(1, -1)
    for i in range(9):
        target_X = np.concatenate((target_X, test_X[0].reshape(1, -1)), axis=0)
        target_y = np.concatenate((target_y, test_y[0].reshape(1, -1)), axis=0)
    z = sess.run(vae.z, {vae.x: target_X, vae.y: target_y})
    first = test_y[0][10]
    step = test_y[0][10]/10
    new_y = test_y[0].copy()

    continuous_y = test_y[0].reshape(1, -1)
    for i in range(9):
        #new_y[10] = first-i*step
        continuous_y = np.concatenate((continuous_y, new_y.reshape(1, -1)), axis=0)

    output_X = sess.run(vae.pi_out, {vae.y: continuous_y, vae.z_in: z})
    output_X = sess.run(vae.pi, {vae.x: target_X, vae.y: continuous_y})
    output_X = sess.run(vae.pi, {vae.x: test_X[0:3], vae.y: test_y[0:3]})

    fig = plt.figure(figsize=(20, 20))

    continuous_imshow10(output_X, fig)

    plt.savefig('reconstruct.png')




