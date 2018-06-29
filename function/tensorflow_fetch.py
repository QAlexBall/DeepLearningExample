import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# fetch
input_1 = tf.constant(3.0)
input_2 = tf.constant(2.0)
input_3 = tf.constant(5.0)
intermed = tf.add(input_2, input_3)
mul = tf.multiply(input_1, intermed)

# feed
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)
    print(sess.run([output], feed_dict={input1: [7.], input2: [2.]}))

