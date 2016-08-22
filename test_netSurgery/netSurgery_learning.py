import numpy as np
import matplotlib.pyplot as plt

import _init_paths
import caffe

from IPython import embed
# first invocation of embed will create an InteractiveShellEmbed instance and call it. Consective calls just call the already created instance.

embed(header='*** Note: we have imported related packages: numpy as np, matplotlib.pylot as plt, caffe ***')

# configure plotting
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Load the net, list its data and params, and filter an example image.
caffe.set_mode_cpu()
net = caffe.Net('../caffe/examples/net_surgery/conv.prototxt', caffe.TEST)
print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
# load image and prepare as a single input batch for Caffe
im = np.array(caffe.io.load_image('../caffe/examples/images/cat_gray.jpg', color=False)).squeeze()
plt.title("original image")
plt.imshow(im)
plt.savefig('./img.png')

plt.axis('off')
plt.savefig('./img_off.png')

# np.newaxis is an alias for None, below we can
# replace with None
im_input = im[np.newaxis, np.newaxis, :, :]
#im_input = im[None, None, :, :]

# (1, 1, 360, 480)
print(im_input.shape)

# modify the data blob input size
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input

print('return to the created ipython instance')
embed()

# helper show filter outputs
def show_filters(net):
    net.forward()
    plt.figure()
    filt_min, filt_max = net.blobs['conv'].data.min(), net.blobs['conv'].data.max()
    for i in range(3):
        plt.subplot(1,4,i+2)
        plt.title("filter #{} output".format(i))
        plt.imshow(net.blobs['conv'].data[0, i], vmin=filt_min, vmax=filt_max)
        plt.savefig('./filter.png')
        plt.tight_layout()
        plt.axis('off')

# filter the image with initial
show_filters(net)

# pick first filter output
conv0 = net.blobs['conv'].data[0, 0]
print("pre-surgery output mean {:.2f}".format(conv0.mean()))
# set first filter bias to 1
net.params['conv'][1].data[0] = 1.
net.forward()
print("post-surgery output mean {:.2f}".format(conv0.mean()))

ksize = net.params['conv'][0].data.shape[2:]
# make Gaussian blur
sigma = 1.
y, x = np.mgrid[-ksize[0]//2 + 1:ksize[0]//2 + 1, -ksize[1]//2 + 1:ksize[1]//2 + 1]
g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
gaussian = (g / g.sum()).astype(np.float32)
net.params['conv'][0].data[0] = gaussian
# make Sobel operator for edge detection
net.params['conv'][0].data[1:] = 0.
sobel = np.array((-1, -2, -1, 0, 0, 0, 1, 2, 1), dtype=np.float32).reshape((3,3))
net.params['conv'][0].data[1, 0, 1:-1, 1:-1] = sobel  # horizontal
net.params['conv'][0].data[2, 0, 1:-1, 1:-1] = sobel.T  # vertical
show_filters(net)

print('return to the created ipython instance')
embed()
#sub example 2: Casting a Classifier into a Fully Convolutional Network


# ! precceds in shell command means executing shell command in python script
# below is the command coparing the difference of the two files

# run this command in ipython enevironment

#!diff ../caffe/examples/net_surgery/bvlc_caffenet_full_conv.prototxt ../caffe/models/bvlc_reference_caffenet/deploy.prototxt


# Load the original network and extract the fully connected layers' parameters.
net = caffe.Net('../caffe/models/bvlc_reference_caffenet/deploy.prototxt',
                        '../caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                                        caffe.TEST)
params = ['fc6', 'fc7', 'fc8']
# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

for fc in params:
        print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

# Load the fully convolutional network to transplant the parameters.
net_full_conv = caffe.Net('../caffe/examples/net_surgery/bvlc_caffenet_full_conv.prototxt',
                                  '../caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                                                            caffe.TEST)
params_full_conv = ['fc6-conv', 'fc7-conv', 'fc8-conv']
# conv_params = {name: (weights, biases)}
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

# fc6-conv weights are (4096, 256, 6, 6) dimensional and biases are (4096, ) dimensional.
# note that convolution weights are arranged in output * input * height * width.
# we roll the flat inner product vectors into channel * height * width

for conv in params_full_conv:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]

net_full_conv.save('../caffe/examples/net_surgery/bvlc_caffenet_full_conv.caffemodel')


print('start a new example')
# another example
# import _init_paths
# import caffe

# import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline

# load input and configure preprocessing
im = caffe.io.load_image('../caffe/examples/images/cat.jpg')
transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
transformer.set_mean('data', np.load('../caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)
# make classification map by forward and print prediction indices at each location
out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
print out['prob'][0].argmax(axis=0)
# show net input and confidence map (probability of the top prediction at each location)
plt.subplot(1, 2, 1)
plt.imshow(transformer.deprocess('data', net_full_conv.blobs['data'].data[0]))
plt.subplot(1, 2, 2)
plt.imshow(out['prob'][0,281])

plt.savefig('./prob.png')

embed()


