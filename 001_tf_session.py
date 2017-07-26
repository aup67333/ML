# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 23:40:33 2017

@author: aup67
"""

import tensorflow as tf
matrix1 = tf.constant([[6,6]])
matrix2 = tf.constant([[3],[3]])

product = tf.matmul(matrix1,matrix2) # 6*3+6*3

#Method 1
sess = tf.Session()
result = sess.run(product)
sess.close()
print (result)
