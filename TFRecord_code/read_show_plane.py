import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile('img_path/plane.jpeg', 'rb').read()

def show(img_data):
    plt.imshow(img_data.eval())
    plt.show()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    print(img_data.eval())
    # plt.imshow(img_data.eval())
    # plt.show()

    resized = tf.image.resize_images(img_data, [500, 500], method=1)
    print(img_data.get_shape())
    # plt.imshow(img_data.eval())
    # plt.show()

    croped = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)
    padded = tf.image.resize_image_with_crop_or_pad(img_data, 3000, 3000)
    # show(resized)
    # show(croped)
    # show(padded)
    central_croped = tf.image.central_crop(img_data, 0.5)
    show(central_croped)


    # img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)
    # encoded_image = tf.image.encode_jpeg(img_data)
    # with tf.gfile.GFile("img_path/encode_plane.jpeg", "wb") as f:
    #    f.write(encoded_image.eval())



