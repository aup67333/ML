# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 13:39:38 2017

@author: aup67
"""
import tensorflow as tf

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

output = tf.multiply(x,y)

sess = tf.Session()
print(sess.run(output, feed_dict={x:[7.],y:[8.]}))

sess.close()