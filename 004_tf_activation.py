import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# fake data
x = np.linspace(-5, 5, 200)     # x data, shape=(100, 1)

# following are popular activation functions
y_relu = tf.nn.relu(x)
y_sigmoid = tf.nn.sigmoid(x)
y_tanh = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)
y_1 = tf.nn.relu6(x)
# y_softmax = tf.nn.softmax(x)  softmax is a special kind of activation function, it is about probability

sess = tf.Session()
y_relu, y_sigmoid, y_tanh, y_softplus,y_1 = sess.run([y_relu, y_sigmoid, y_tanh, y_softplus,y_1])

# plt to visualize these activation function
plt.figure(1, figsize=(8, 6))
plt.subplot(331)
plt.plot(x, y_relu, c='red', label='relu')
#plt.ylim((-1, 10))
plt.legend(loc='best')

plt.subplot(332)
plt.plot(x, y_sigmoid, c='red', label='sigmoid')
#plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(333)
plt.plot(x, y_tanh, c='red', label='tanh')
#plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(334)
plt.plot(x, y_softplus, c='red', label='softplus')
#plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.subplot(335)
plt.plot(x, y_1 , c='red', label='relu6')
plt.legend(loc='best')

plt.show()