# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 23:47:50 2017

@author: aup67
"""

import tensorflow as tf

state = tf.Variable(0)
value = tf.constant(1)

new_value = tf.add(state,value)
updata = tf.assign(state,new_value)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for a in range(5):
    result = sess.run(updata)
    print(sess.run(state))
    
sess.close()