from __future__ import print_function
import warnings
import logging, os

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

# define tensor constants
a = tf.constant(2)
b = tf.constant(3)
c = tf.constant(5)

# Various tensor operations
# Note: Tensor also support python operators (+, *, ...)
add = tf.add(a, b)
sub = tf.subtract(a, b)
mul = tf.multiply(a, b)
div = tf.divide(a, b)

# Access tensors value
print("add = ", add.numpy())
print("sub = ", sub.numpy())
print("mul = ", mul.numpy())
print("div = ", div.numpy())

# Some more operations.
mean = tf.reduce_mean([a, b, c])
sum = tf.reduce_sum([a, b, c])

# Access tensors value.
print("mean =", mean.numpy())
print("sum =", sum.numpy())

# Matrix multiplications.
matrix1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
matrix2 = tf.constant([[5.0, 6.0], [7.0, 8.0]])
product = tf.matmul(matrix1, matrix2)
print(product)
print(product.numpy())
