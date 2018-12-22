from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from voxel import voxel2obj
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.examples.tutorials.mnist import input_data

from timeit import default_timer as timer
from numba import vectorize
from numba import cuda

total_time_start = timer()
batch_size = 10  # Number of samples in each batch
epoch_num = 5  # Number of epochs to train the network
lr = 0.001  # Learning rate

#@vectorize(["float32(float32,float32)"], target='cuda')
def interpolationBetnLatentSpace(z1, z2):
    mx = 100
    mn = 0
    for t in range(mn,mx,1):
        new_z = (1 - t) * z1 + t * z2
    return new_z

def resize_batch(imgs):
    # A function to resize a batch of MNIST images to (32, 32)
    # Args:
    #   imgs: a numpy array of size [batch_size, 28 X 28].
    # Returns:
    #   a numpy array of size [batch_size, 32, 32].
    imgs = imgs.reshape((-1, 32, 32, 32, 1))

    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32, 32))
    print(resized_imgs.shape)
    return resized_imgs


def loadfile():
    input_file = np.array([])
    for i in range(21, 71):
        v = np.loadtxt('/home/tansinjahan/Desktop/testLinear/Volume of shapes/MyTestFile' + str(i) + '.txt')
        image_matrix = np.reshape(v, (32, 32, 32)).astype(np.float32)
        z, x, y = image_matrix.nonzero()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, -z, zdir='z', c='red')
        plt.savefig('input_data/demo' + str(i) + '.png')
        input_file = np.append(input_file, image_matrix)

    input_file = np.reshape(input_file, (50, 32 * 32 * 32))
    print("This is the shape of input for 50 shape", input_file.shape)
    return input_file


# read dataset
input_file = loadfile()  # load 50 chairs as volume with shape [50,32768]


def autoencoder(inputs):
    # encoder
    # 32 x 32 x 32 x 1   -> 16 x 16 x 16 x 32
    # 16 x 16 x 16 x 32  ->  8 x 8 x 8 x 16
    # 8 x 8 x 8 x 16    ->  2 x 2 x 2 x 8
    net = lays.conv3d(inputs, 32, [5, 5, 5], stride=2, padding='SAME')
    net = lays.conv3d(net, 16, [5, 5, 5], stride=2, padding='SAME')
    print(tf.shape(net))
    net = lays.conv3d(net, 8, [5, 5, 5], stride=4, padding='SAME')

    latent_space = net
    # decoder
    # 2 x 2 x 2 x 8   ->  8 x 8 x 8 x 16
    # 8 x 8 x 8 x 16  ->  16 x 16 x 16 x 32
    # 16 x 16 x 16 x 32  ->  32 x 32 x 32 x 1
    net = lays.conv3d_transpose(net, 16, [5, 5, 5], stride=4, padding='SAME')
    net = lays.conv3d_transpose(net, 32, [5, 5, 5], stride=2, padding='SAME')
    net = lays.conv3d_transpose(net, 1, [5, 5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
    return latent_space, net


def next_batch(next_batch_array, batchsize, offset):
    rowStart = offset * batchsize
    rowEnd = (rowStart + batchsize) - 1
    return next_batch_array[rowStart:rowEnd, :]



# calculate the number of batches per epoch


batch_per_ep = input_file.shape[0] // batch_size  # batch per epoch will be 5 [input total = 50 divided by batch-size = 10 ]

ae_inputs = tf.placeholder(tf.float32, (None, 32, 32, 32, 1))  # input to the network (MNIST images)

l_space,ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network

# calculate the loss and optimize the network
loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # calculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# initialize the network
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch_num):  # epochs loop
        next_batch_array = input_file  # copy of input file to use for fetching next batch from input array
        for batch_n in range(batch_per_ep):  # batches loop
            batch_img = next_batch(next_batch_array, batch_size, batch_n)  # read a batch
            batch_img = resize_batch(batch_img)  # reshape the images to (32, 32)
            _, c = sess.run([train_op, loss], feed_dict={ae_inputs: batch_img})
            print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))


# test the trained network
    #batch_img = next_batch(input_file,1,0)

    # Test for the first input volume shape
    batch_img = input_file[0,:]
    batch_img = resize_batch(batch_img)
    recon_img = sess.run([l_space,ae_outputs], feed_dict={ae_inputs: batch_img})[1]
    l_space1 = sess.run([l_space,ae_outputs], feed_dict={ae_inputs: batch_img})[0]
    print("this is output image type", type(recon_img))
    print(recon_img.shape)

    out = recon_img[0,...,0]

    out = np.reshape(out, (32, 32, 32)).astype(np.float32)
    print(out)
    # plot the reconstructed images and their ground truths (inputs)

    z, x, y = out.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.savefig('reconstruct1.png')

    # Test for the second input volume shape
    batch_img = input_file[1, :]
    batch_img = resize_batch(batch_img)
    recon_img = sess.run([l_space,ae_outputs], feed_dict={ae_inputs: batch_img})[1]
    l_space2 = sess.run([l_space, ae_outputs], feed_dict={ae_inputs: batch_img})[0]
    print("this is output image type", type(recon_img))
    print(recon_img.shape)

    out = recon_img[0, ..., 0]

    out = np.reshape(out, (32, 32, 32)).astype(np.float32)
    print(out)
    # plot the reconstructed images and their ground truths (inputs)

    z, x, y = out.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='blue')
    plt.savefig('reconstruct2.png')


    start = timer()
    print("l_space1" , l_space1.shape)
    print("l_space2" , l_space2.shape)
    l_space1 = np.reshape(l_space1, (8,8))
    l_space2 = np.reshape(l_space2, (8,8))
   
    new_z = interpolationBetnLatentSpace(l_space1,l_space2)
    print("new_z", new_z)
    vectoradd_time = timer() - start
    print("Vector Add took %f seconds in CPU:" , vectoradd_time)

    total_time_end = timer()
    total_time = total_time_start - total_time_end
    print("total time CPU:", "%.2f" % total_time, "s")
    

