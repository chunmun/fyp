import theano as th
from theano import tensor as T
from theano.tensor.nnet import conv

import numpy as np

"""
rng = np.random.RandomState(0)

# instantiate 4D tensor for input
input  = T.tensor4(name='input')

# initialize shared variable for weights
w_shp = (2, 3, 9, 9)
w_bound = np.sqrt(3 * 9 * 9)
W = th.shared(
        np.asarray(
            rng.uniform(
                low = -1.0 / w_bound,
                high = 1.0 / w_bound,
                size = w_shp),
            dtype = input.dtype), name = 'W')

# initialize shared variable for bias (1D tensor) with random values
# IMPORTANT: biases are usually initialized to zero. However in this
# particular application, we simply apply the convolutional layer to
# an image without learning the parameters. We therefor initialize
# them to random values to 'simulate' learning
b_shp = (2,)
b = th.shared(
        np.asarray(
            rng.uniform(
                low = -0.5,
                high = 0.5,
                size = b_shp),
            dtype = input.dtype), name = 'b')

# build symbolic expression that computes the convolution of input wiht filters
# in w
conv_out = conv.conv2d(input, W)

# build symbolic expression to add bias and apply activation function,
# A few words on 'dimshuffle' :
#  'dumshuffle' is a powerful tool in reshaping a tensor;
#  what it allows you to do is to shuffle dimension around
#  but also to insert new ones along which the tensor will be
#  broadcastable:
#  dimshuffle('x', 2, 'x', 0, 1)
#  This will work on 3d tensors with no broadcastable
#  dimensions. The first dimension will be broadcastable,
#  then we will have the third dimension of hte input tensor as
#  the second of the resulting tensor, etc. If the tensor has
#  shape (20, 30, 40), the resulting tensor will have dimensions
#  (1, 40, 1, 20, 30). (AxBxC tensor is mapped to 1xCx1xAxB tensor)
#   More examples:
#    dimshuffle('x') -> make a 0d (scalar) into a 1d vector
#    dimshuffle(0, 1) -> identity
#    dimshuffle(1, 0) -> inverts the first and second dimensions
#    dimshuffle('x', 0) -> make a row out of a 1d vector (N to 1xN)
#    dimshuffle(0, 'x') -> make a column out of a 1d vector (N to Nx1)
#    dimshuffle(2, 0, 1) -> AxBxC to CxAxB
#    dimshuffle(0, 'x', 1) -> AxB to Ax1xB
#    dimshuffle(1, 'x', 0) -> AxB to Bx1xA

output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

# create theano function to compute filtered images
f = th.function([input], output)

import matplotlib.pyplot as plt
from PIL import Image

# open random image of dimensions
img = Image.open('/home/chunmun/Pictures/birthday.jpg')
# dimensions are (height, width, channel)
img = np.asarray(img, dtype='float32') / 256

# put image in 4D tensor of shape (1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, 3, 320, 320)
filtered_img = f(img_)

# plot original image and first and second componenets of output
plt.subplot(1, 3, 1); plt.axis('off'); plt.imshow(img)
plt.gray();
# recall that the convOp output (filtered image) is actually a 'minibatch'
# of size 1 here, so we take index 0 in the first dimensionL
plt.subplot(1, 3, 2); plt.axis('off'); plt.imshow(filtered_img[0, 0, :, :,])
plt.subplot(1, 3, 3); plt.axis('off'); plt.imshow(filtered_img[0, 1, :, :,])
plt.show()
"""
from theano.tensor.signal import downsample
input = T.dtensor4('input')
maxpool_shape = (2, 2)
pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=True)
f = th.function([input], pool_out)

invals = np.random.RandomState(1).rand(3, 2, 5, 5)
print('With ignore_border set to True:')
print('invals[0, 0, :, :] = \n', invals[0, 0, :, :])
print('output[0, 0, :, :] = \n', f(invals)[0, 0, :, :])

pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=False)
f = th.function([input], pool_out)
print('With ignore_border set to False:')
print('invals[0, 0, :, :] = \n', invals[0, 0, :, :])
print('output[0, 0, :, :] = \n', f(invals)[0, 0, :, :])
