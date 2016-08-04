import numpy as np
import cv2
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original')
mnist_X, mnist_y = shuffle(mnist.data, mnist.target.astype('int32'))

mnist_X = mnist_X/255.0

train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y, test_size=0.2)

img = train_X[0]

img = img.reshape(-1, 28)

fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)

ax.imshow(img)

center = (img.shape[0]*0.5, img.shape[1]*0.5)

size = (img.shape[0], img.shape[1])

angle = 45.0

scale = 1.0

rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

img_rot = cv2.warpAffine(img, rotation_matrix, size)

ax2 = fig.add_subplot(2, 2, 2)
ax2.imshow(img_rot)

plt.show()



