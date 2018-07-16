"""
Following a tutorial on how to build a simple NN with tensorflow
"""

import numpy as np
import tensorflow as tf

def sigmoid(x):
	return 1/(1+np.exp(-x))

n_input=3
n_hidden=2
n_output=1

w = { 
	'h1' : tf.Variable(tf.ones([n_input,n_hidden]),name="h1"),
	'out' : tf.Variable(tf.ones([n_hidden,n_output]),name="out")
}

b = {
	'b1' : tf.Variable(tf.zeros([n_hidden])),
	'bout' : tf.Variable(tf.zeros([n_output]))
}

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])

l1 = tf.add(tf.matmul(x,w['h1']),b['b1'])
l1_act = tf.sigmoid(l1)

out = tf.add(tf.matmul(l1_act,w['out']),b['bout'])
out_act = tf.sigmoid(out)

cost = tf.reduce_mean(tf.abs(tf.subtract(out_act,y)))
train_step = tf.train.AdadeltaOptimizer(learning_rate=1.0).minimize(cost)

x_raw = np.array([[1,2,3]])
y_raw = np.array([3])
y_raw = np.reshape(y_raw,(1,1))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	pred = out_act.eval({x: x_raw})
	print(pred)
	g = w['out']

	for epoch in range(1000000):
		_,c,o,b=sess.run([train_step,cost,out_act,g],feed_dict={x:x_raw,y:y_raw})
		if epoch == 0:
			print(c,o[0][0],b[0][0])
	print(c,o[0][0],b[0][0])
	#print(sess.run(y))
