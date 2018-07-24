import tensorflow as tf

reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(
    ["path/output.tfrecords"])

_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw' : tf.FixedLenFeature([], tf.string),
        'pixels' : tf.FixedLenFeature([], tf.int64),
        'labels' : tf.FixedLenFeature([], tf.int64),
    })

# tg.recode_raw可以将字符串解析成图像对应的像素数组
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['labels'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()

# 启动多线程处理数据
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])
    print(image, label, pixel)
