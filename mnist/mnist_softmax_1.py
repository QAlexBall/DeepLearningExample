# 04/08/2017
# QAlexBall
#
import os

from tensorflow.python.training import saver

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
# download MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("Done!")

# graph'x'
x = tf.placeholder(tf.float32, [None, 784])

# Weights
W = tf.Variable(tf.zeros([784, 10]))

# bias
b = tf.Variable(tf.zeros([10]))

# probability
y = tf.nn.softmax(tf.matmul(x, W) + b)

# In order to calculate the cross entropy
y_ = tf.placeholder(tf.float32, [None, 10])

# calculate
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# gradient descent algorithm
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

# initlize variables
init = tf.global_variables_initializer()

# define saver for save module
saver = tf.train.Saver()

# start model
with tf.Session() as sess:
    sess.run(init)
    # train model
    for i in range(2000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    save_path = saver.save(sess, "data1/model.ckpt")
    print("Model saved in file: %s" % save_path)
    write = tf.summary.FileWriter("/QAlexBall/tensorflow_learning_code/mnist/path/to/log", tf.get_default_graph())
    write.close()

