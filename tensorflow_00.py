# -*- coding: utf-8 -*-


"""
From Lession 1.8 of TensorFlow Basics "Session 1: Introduction to TensorFlow"
https://youtu.be/uO3CMMT459w
TensorFlow Basics | Kadenze

Aug 16, 2016

"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as mp
from skimage import data

demo = {
	"guassian"     : True,
	"convolution"  : True,
	"gabor"        : True,
	"placeholders" : True
}

msg=\
"""TensorFlow Workbook
*****************************************************************************\n
In numpy we could use linear space function np.linspace(-3.0, 3.0, 100)
"""
print(msg)

x_np = np.linspace(-3.0, 3.0, 100)

print("""here are the properties of the np.linspace object:""")
print("""\nvalues:""")
print(x_np)
print("""\nshape: """+str(x_np.shape))
print("""dtype: """+str(x_np.dtype))

msg=\
"""as you can see we have a shape(size and dimension) and a datatype. But if we create\n
a tensor version of this:\n
"""

tf_x = tf.linspace(-3.0, 3.0, 100)

print("""we get the following object:""")

print("""tf object(Tensor): """+str(tf_x))

msg=\
"""
TF has not computed these values yet, instead it is the operation which is added to the
default computational graph.

Think about it: a way to link functions together in a graph form allows for unqiue 
(and editable) traverals of complex mathmatical operations (that use a GPU) :)

To compute anything in TF we need a Session. The session evaluates the graph!
"""
print(msg)

msg=\
"""
Please READ my code!
"""
print(msg)
# delete the msg=\ and the print lines to keep the comments

tf_s = tf.Session() 				# create session
cpt_x = tf_s.run(tf_x) 				# execute the Tensor 
# OR
cpt_x = tf_x.eval(session=tf_s)		# evaluate the Tensor

print(cpt_x)

#gph_0 = tf.Graph()					# create new graph

print(tf_x.get_shape()) 			# get the shape
print(tf_x.get_shape().as_list())	# get as list


msg=\
"""
 ██████╗  █████╗ ██╗   ██╗███████╗███████╗██╗ █████╗ ███╗   ██╗
██╔════╝ ██╔══██╗██║   ██║██╔════╝██╔════╝██║██╔══██╗████╗  ██║
██║  ███╗███████║██║   ██║███████╗███████╗██║███████║██╔██╗ ██║
██║   ██║██╔══██║██║   ██║╚════██║╚════██║██║██╔══██║██║╚██╗██║
╚██████╔╝██║  ██║╚██████╔╝███████║███████║██║██║  ██║██║ ╚████║
 ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝
                                                               
 ██████╗██╗   ██╗██████╗ ██╗   ██╗███████╗     
██╔════╝██║   ██║██╔══██╗██║   ██║██╔════╝     
██║     ██║   ██║██████╔╝██║   ██║█████╗       
██║     ██║   ██║██╔══██╗╚██╗ ██╔╝██╔══╝       
╚██████╗╚██████╔╝██║  ██║ ╚████╔╝ ███████╗     
 ╚═════╝ ╚═════╝ ╚═╝  ╚═╝  ╚═══╝  ╚══════╝     
"""
print(msg)

mean = 0
sigma = 1.0
# https://bit.ly/2GRurlp
# *negative might need to be neg for other version of TF
tf_z=(tf.exp(tf.negative(tf.pow(tf_x-mean,2.0) /
	(2.0 * tf.pow(sigma,2.0)))) *
	(1.0/(sigma*tf.sqrt(2.0*3.1415))))
# and let's compute it

tf_r = tf_z.eval(session=tf_s)			# evaluate the Tensor

# matplotlib stuff

"""
Plot the 
"""

# matplotlib guassian curve (IMAGE #1)
if demo['guassian']:
	mp.plot(tf_r)
	mp.show()

tf_ksize = tf_z.get_shape().as_list()[0]

print("""ksize: """+str(tf_ksize))

# to get 2-d gaussian (take gaussian along one dim (x) and multiply by another dim(y) )
tf_z_2d = tf.matmul(tf.reshape(tf_z,[tf_ksize,1]),tf.reshape(tf_z,[1,tf_ksize]))

# matplotlib to show the result (IMAGE #2)
if demo['guassian']:
	mp.imshow(tf_z_2d.eval(session=tf_s))
	mp.show()

"""
From Lession 1.9 of TensorFlow Basics "Session 1: Introduction to TensorFlow"
https://youtu.be/ETdaP_bBNWc
Understanding Convolution with TensorFlow | Kadenze

Jul 27, 2016

"""
msg=\
"""
 ██████╗ ██████╗ ███╗   ██╗██╗   ██╗ ██████╗ ██╗     ██╗   ██╗████████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██╔═══██╗████╗  ██║██║   ██║██╔═══██╗██║     ██║   ██║╚══██╔══╝██║██╔═══██╗████╗  ██║
██║     ██║   ██║██╔██╗ ██║██║   ██║██║   ██║██║     ██║   ██║   ██║   ██║██║   ██║██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║╚██╗ ██╔╝██║   ██║██║     ██║   ██║   ██║   ██║██║   ██║██║╚██╗██║
╚██████╗╚██████╔╝██║ ╚████║ ╚████╔╝ ╚██████╔╝███████╗╚██████╔╝   ██║   ██║╚██████╔╝██║ ╚████║
 ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝  ╚═══╝   ╚═════╝ ╚══════╝ ╚═════╝    ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
"""
print(msg)

msg=\
"""
Convolulation - filtering information: we will filter an image using the gaussian as a lens
to view our image data through.

Think about it: viewing our image in a way in which we apply a filter such that the
center is fully on (1) and all other points are less than 1 (fractions) as they fall away
from that center...  "Keeps a lot of the center and then keeps less and less as we move away
from the center"... the result is that the image will be blurred
"""
print(msg)

# load "camera guy" image from skimage


img=data.camera().astype(np.float32)

if demo['convolution']:
	mp.imshow(img, cmap='gray')

# matplotlib to show the result (IMAGE #2)
mp.show()

msg=\
"""
Notice that the image is 2-D. We need 4-D for TF.
"""
print(msg)

print(img.shape)

img_4d = tf.reshape(img,[1,img.shape[0],img.shape[1],1])

print("""shape: """+str(img_4d.get_shape().as_list()))

msg=\
"""
The image is now part of the 4-D array where the other dimensions are 'n' and 'c' and are
1 in this case. (1 image and 1 channel)
Image dimensions              = N x H x W x C
Convolution kernel dimensions = Kh x Kw x C x Kn
"""
print(msg)

tf_z_4d = tf.reshape(tf_z_2d, [tf_ksize, tf_ksize, 1, 1])

print("""shape of kernel: """+str(tf_z_4d.get_shape().as_list()))

msg=\
"""
Now we can convolve the guassian with the image. The stride describes how the guassian is
moved across the image as it convolves. We have to convert the result back to
a 2-D image. Here we use the squeeze function from numpy.
"""
print(msg)

tf_stride = [1,1,1,1]			# [n,h,w,c]
#tf_stride = [1,4,4,1]

if demo['convolution']:
	convolved = tf.nn.conv2d(img_4d,tf_z_4d,strides=tf_stride,padding='SAME')
	tf_res = convolved.eval(session=tf_s)
	mp.imshow(np.squeeze(tf_res),cmap='gray')
	mp.show()

msg=\
"""
Or we can show the exact dims we want to visualize. Here also I'm not using a cmap and
you'll notice that it's not gray scale becuase I didn't specify that.
"""
print(msg)

if demo['convolution']:
	mp.imshow(tf_res[0,:,:,0])
	mp.show()

msg=\
"""
 ██████╗  █████╗ ██████╗  ██████╗ ██████╗         
██╔════╝ ██╔══██╗██╔══██╗██╔═══██╗██╔══██╗        
██║  ███╗███████║██████╔╝██║   ██║██████╔╝        
██║   ██║██╔══██║██╔══██╗██║   ██║██╔══██╗        
╚██████╔╝██║  ██║██████╔╝╚██████╔╝██║  ██║        
 ╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝        
                                                  
██╗  ██╗███████╗██████╗ ███╗   ██╗███████╗██╗     
██║ ██╔╝██╔════╝██╔══██╗████╗  ██║██╔════╝██║     
█████╔╝ █████╗  ██████╔╝██╔██╗ ██║█████╗  ██║     
██╔═██╗ ██╔══╝  ██╔══██╗██║╚██╗██║██╔══╝  ██║     
██║  ██╗███████╗██║  ██║██║ ╚████║███████╗███████╗
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝
"""
print(msg)

tf_xs = tf.linspace(-3.0, 3.0, tf_ksize)
tf_ys = tf.sin(tf_xs)

if demo['gabor']:
	mp.plot(tf_ys.eval(session=tf_s))
	mp.show()

# sine wave aligned with y-axis and repeated across the matrix
tf_ys = tf.reshape(tf_ys, [tf_ksize,1])
ones = tf.ones((1, tf_ksize))
wave = tf.matmul(tf_ys, ones)

if demo['gabor']:
	mp.imshow(wave.eval(session=tf_s))
	mp.show()

# sine wave aligned with x-axis and repeated across the matrix
tf_ys = tf.reshape(tf_ys, [1,tf_ksize])
ones = tf.ones((tf_ksize,1))
wave = tf.matmul(ones,tf_ys)

if demo['gabor']:
	mp.imshow(wave.eval(session=tf_s))
	mp.show()

gabor = tf.multiply(wave, tf_z_2d)

if demo['gabor']:
	mp.imshow(gabor.eval(session=tf_s))
	mp.show()

msg=\
"""
██████╗ ██╗      █████╗  ██████╗███████╗██╗  ██╗ ██████╗ ██╗     ██████╗ ███████╗██████╗ ███████╗
██╔══██╗██║     ██╔══██╗██╔════╝██╔════╝██║  ██║██╔═══██╗██║     ██╔══██╗██╔════╝██╔══██╗██╔════╝
██████╔╝██║     ███████║██║     █████╗  ███████║██║   ██║██║     ██║  ██║█████╗  ██████╔╝███████╗
██╔═══╝ ██║     ██╔══██║██║     ██╔══╝  ██╔══██║██║   ██║██║     ██║  ██║██╔══╝  ██╔══██╗╚════██║
██║     ███████╗██║  ██║╚██████╗███████╗██║  ██║╚██████╔╝███████╗██████╔╝███████╗██║  ██║███████║
╚═╝     ╚══════╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝
"""
print(msg)

msg=\
"""
We know we need something here but we don't know what yet! Generally the input and output of the
network. Let's rewrite with placeholders.
"""
print(msg)

# shape=[None,None] -> let this dim be any possible value
tf_input = tf.placeholder(tf.float32,shape=[None,None],name='tf_input')
tf_input_3d = tf.expand_dims(tf_input,2) #expand on the 2nd axis ([None,None]->[None,None,1])
print("""3d shape: """+str(tf_input_3d.get_shape().as_list()))
tf_input_4d = tf.expand_dims(tf_input_3d,0)
print("""4d shape: """+str(tf_input_4d.get_shape().as_list()))

tf_mean = tf.placeholder(tf.float32,name='mean')
tf_sigma = tf.placeholder(tf.float32,name='sigma')
tf_ksize = tf.placeholder(tf.int32,name='ksize')

tf_x = tf.linspace(-3.0,3.0,tf_ksize)

tf_z=(tf.exp(tf.negative(tf.pow(tf_x-tf_mean,2.0) /
	(2.0 * tf.pow(tf_sigma,2.0)))) *
	(1.0/(tf_sigma*tf.sqrt(2.0*3.1415))))

tf_z_2d = tf.matmul(
	tf.reshape(tf_z,tf.stack([tf_ksize,1])),
	tf.reshape(tf_z,tf.stack([1,tf_ksize])))

tf_ys = tf.sin(tf_x)
tf_ys = tf.reshape(tf_ys,tf.stack([tf_ksize,1]))

ones = tf.ones(tf.stack([1,tf_ksize]))
wave = tf.matmul(tf_ys,ones)

gabor = tf.multiply(wave,tf_z_2d)
gabor_4d = tf.reshape(gabor,tf.stack([tf_ksize,tf_ksize,1,1]))

convolved = tf.nn.conv2d(img_4d,gabor_4d,strides=tf_stride,padding='SAME',name='convolved')
convolved_img = convolved[0,:,:,0]

# to get a list of all the tensors -->
#print([n.name for n in tf.get_default_graph().as_graph_def().node])

res = convolved_img.eval(session=tf_s,feed_dict={
	tf_input:data.camera(),
	tf_mean:0.0,
	tf_sigma:0.5,
	tf_ksize:4})

mp.imshow(res,cmap='gray')
mp.show()