#
#
#
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# create a constant op, generate a 1*2 matrix_1. this op called node
# add to defualt graph
#
# the return value of the constructor represents the return value of the constant op

matrix_1 = tf.constant([[4., 3.]])

# create another constant op, generate a 2*1 matrix_2.

matrix_2 = tf.constant([[5.], [2.]])

# create a (mat rix) (mul tiplication) (mat mul) op, use 'matrix_1' and 'matrix_2' as output.
# the return value 'product' represents matmul's result.

product = tf.matmul(matrix_1, matrix_2)

# stat the default graph

with tf.Session() as sess:
    result = sess.run([product])
    print (result)

'''
sess = tf.Session()
# call sess 'run' method to execute matmul op, pass 'product' as a parameter to the method
# mentioned above, 'product represents the output.jpeg of matmul op,passing into it is for show,
# we want retrive(取回）the output.jpeg of matmul op
#
result = sess.run(product)
print (result)
sess.close()
'''
