import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# mnist数据集相关的常数
INPUT_NODE = 784    # 输入层节点数。对于MNIST数据集相当于图片像素。
OUTPUT_NODE = 10    # 输出层节点数。 等于类别数目。

# 配置神经网络的参数。
LAYER1_NODE = 500   # 隐藏层节点数。 500个节点。
BATCH_SIZE = 100    # 一个训练batch中的训练数据个数。数字越小时，越接近
                    # 随机梯度下降；数字 越大时越接近梯度下降。
LEARNING_RATE_BASE = 0.8    # 基础的学习率。
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率。
REGULARIZATION_RATE = 0.0001    # 描述模型复杂度的正则化项在损失函数中的系数。
TRAINING_STEPS = 30000          # 训练轮数。
MOVING_AVERGE_DECAY= 0.99       # 滑动平均衰减率。

# 一个辅助函数， 给定神经网络的输入和所有参数，计算神经网络的前向传播结果。在这里
# 定义了一个使用ReLU激活函数的三层全连接神经网络。通过加入隐藏层实现了多层网络结构，
# 通过ReLU激活函数实现了去线性化。在这个函数中也支持传入用于计算参数平均值的类，
# 这样方便在测试时使用滑动平均模型。
def inference(input_tensor, avg_class, weights1, biases1,
              weights2, biases2):
    # 当没有提供滑动平均类时，直接使用参数当前的取值。
    if avg_class == None:
        # 计算隐藏层的前向传播结果，这里使用了Relu激活函数。
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)

        # 计算输出层的前向传播结果。因为在计算损失函数时会一并计算softmax函数，
        # 所以这里不需要加入激活函数。而且不加入softmax不会影响预测结果。因为预测时，
        # 使用的是不同类别对应节点输出值的相对大小，有没有softmax层对最后类结果的
        # 计算没有影响。于是在计算整个神经网络的前向传播时可以不加入最后的softmax层。
        return tf.matmul(layer1, weights2) + biases2

    else:
        # 首先使用avg_class.average函数来计算得出变量的滑动平均值，
        # 然后再计算相应的神经网络向前传播结果。
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

# 训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

     # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输出层的参数。
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算在当前参数下神经网络前向传播的结果。这里给出的用于计算滑动平均的类为None，
    # 所以函数不会使用参数的滑动平均值。
    y = inference(x, None, weights1, biases1, weights2, biases2)


    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERGE_DECAY, global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    regularization = regularizer(weights1) + regularizer(weights2)

    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_example / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )



    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy"
                      "model is %g" % (TRAINING_STEPS, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

    test_acc = sess.run(accuracy, feed_dict=test_feed)
    print("After %d training step(s), test accuracy using average"
          "model is %g" %  (TRAINING_STEPS, test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets("mnist/data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()





