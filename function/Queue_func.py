import tensorflow as tf

# 创建一个先进先出的队列， 指定队列中最多可以保存两个元素，并指定类型为整数。
q = tf.FIFOQueue(2, "int32")
# 使用enqueue_many函数来初始化队列中的元素。和变量初始化类似，在使用队列之前
# 需要明确的调用这个初始化的过程。
init = q.enqueue_many(([0, 10],))
