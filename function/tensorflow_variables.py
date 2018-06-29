import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
# create a variables, initialize to scalar 0.

state = tf.Variable(0, name = "counter")

#create a op, its effect is to increase the state 1.

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# after starting the graph,the variable must be initialized by 'initialize' (init)op
# add an 'initialize'op to the graph, firstly.

init_op = tf.global_variables_initializer()# initialize_all_variables is deprecated(弃用),use 'tf.global_variables_initializer' instead.

# start graph, run op
with tf.Session() as sess:
    # run 'init'op
    sess.run(init_op)
    # print the initial value of 'state'
    print (sess.run(state))
    # run op, update 'state',and print 'state'
    for _ in range(3):
        sess.run(update)
        print (sess.run(state))



