"""CNN to colorize grayscale images."""

from keras.applications.vgg16 import VGG16
from keras.models import Model, model_from_json
from keras.layers import Input, Add
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, concatenate
from keras.preprocessing import image
from keras import backend as K

import matplotlib.pyplot as plt
import pdb
import os
import tensorflow as tf
import numpy as np
import sys
import cv2


def dist(x, y):
    """Euclidean distance loss."""
    return K.sqrt(K.sum(K.square(y - x)))


def blur(img, kernel_size):
    """Apply gaussian blurring to the image."""
    kernel_x = cv2.getGaussianKernel(kernel_size, sigma=4)
    kernel_y = cv2.getGaussianKernel(kernel_size, sigma=4)
    kernel = kernel_x*np.transpose(kernel_y)
    kernel = np.dstack([kernel, kernel, kernel])
    kernel = tf.convert_to_tensor(kernel, dtype=tf.float32)
    kernel = tf.expand_dims(kernel, -1)

    filtered_image = tf.nn.depthwise_conv2d(img, kernel,
                                            [1, 1, 1, 1], 'SAME')

    return filtered_image


def rgb2uv(img):
    """Convert rgb image to yuv space."""
    rgb_to_yuv_kernel = [[0.299, -0.14714119, 0.61497538],
                         [0.587, -0.28886916, -0.51496512],
                         [0.114, 0.43601035, -0.10001026]]
    kernel = tf.convert_to_tensor(rgb_to_yuv_kernel, dtype=tf.float32)
    ndims = img.get_shape().ndims

    img_yuv = tf.tensordot(img, kernel, axes=[[ndims - 1], [0]])
    return img_yuv[:, :, :, 1:]


def blur_uv_loss(rgb, inferred_rgb):
    """Calculate custom loss."""
    uv = rgb2uv(rgb)
    uv_blur0 = rgb2uv(blur(rgb, 3))
    uv_blur1 = rgb2uv(blur(rgb, 5))

    inferred_uv = rgb2uv(inferred_rgb)
    inferred_uv_blur0 = rgb2uv(blur(inferred_rgb, 3))
    inferred_uv_blur1 = rgb2uv(blur(inferred_rgb, 5))
    loss = (dist(inferred_uv, uv) +
            dist(inferred_uv_blur0, uv_blur0) +
            dist(inferred_uv_blur1, uv_blur1))/3
    return loss


def batch_generator(X, y, batch_size):
    """Generate batches for training."""
    Xb_shape = list(X.shape)
    Xb_shape[0] = batch_size
    X_batch = np.zeros(tuple(Xb_shape))

    yb_shape = list(y.shape)
    yb_shape[0] = batch_size
    y_batch = np.zeros(tuple(yb_shape))

    while True:
        for i in range(batch_size):
            index = np.random.randint(0, high=X.shape[0])
            X_batch[i] = X[index]
            y_batch[i] = y[index]
        yield X_batch, y_batch


def define_model():
    """CNN model definition."""
    input_image = Input(shape=(224, 224, 1))
    img = concatenate([input_image, input_image], axis=-1)
    img = concatenate([img, input_image], axis=-1)

    vgg_model = VGG16(include_top=False, weights="imagenet", input_tensor=img)  # Y

    for layer in vgg_model.layers:
        layer.trainable = False

    x = BatchNormalization(name="bn1")(vgg_model.get_layer(name="block4_conv3").output)
    x = Conv2D(256, (1, 1), activation="relu", name="conv1")(x)
    up1 = UpSampling2D(size=(2, 2))(x)
    x1 = BatchNormalization(name="bn2")(vgg_model.get_layer(name="block3_conv3").output)
    x = Add()([x1, up1])
    x = Conv2D(128, (1, 1), activation="relu", name="conv2")(x)
    up2 = UpSampling2D(size=(2, 2))(x)
    x2 = BatchNormalization(name="bn3")(vgg_model.get_layer(name="block2_conv2").output)
    x = Add()([x2, up2])
    x = Conv2D(64, (1, 1), activation="relu", name="conv3")(x)
    up3 = UpSampling2D(size=(2, 2))(x)
    x3 = BatchNormalization(name="bn4")(vgg_model.get_layer(name="block1_conv2").output)
    x = Add()([x3, up3])
    x = Conv2D(3, (1, 1), activation="relu", name="conv4")(x)
    uv = Conv2D(2, (1, 1), activation="relu", name="conv5")(x)  # UV output
    pred_img = concatenate([input_image, uv], axis=-1)

    model = Model(inputs=input_image, outputs=pred_img)
    return model


if __name__ == "__main__":
    input_path = "/home/chrizandr/colorize/input/"
    output_path = "/home/chrizandr/colorize/output/"
    files = set(os.listdir(input_path)).intersection(set(os.listdir(output_path)))

    X = np.empty((len(files), 224, 224, 1))
    y = np.empty((len(files), 224, 224, 3))
    for i, f in enumerate(files):
        img = cv2.imread(input_path+f, 0)
        x = cv2.resize(img, (224, 224))
        x = np.expand_dims(x,-1)
        X[i] = x

        pred_img = cv2.imread(output_path+f)
        try:
           x = cv2.resize(pred_img, (224, 224))
        except:
           print(f)
        y[i] = x

    if len(sys.argv) == 2:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("model.h5")
        print("Loaded model from disk")
    else:
        model = define_model()

    model.compile(loss=blur_uv_loss,
                  optimizer='adam',
                  metrics=['accuracy'])
    #model.fit_generator(batch_generator(X, y, 3), steps_per_epoch=30, epochs=1500)

    test_path = "test/"
    files = os.listdir(test_path)
    timages = np.zeros((len(files), 224, 224, 1))
    for i, f in enumerate(files):
        img = cv2.imread(test_path+f, 0)
        x = cv2.resize(img, (224, 224))
        x = np.expand_dims(x, -1)
        timages[i] = x


    pimages = model.predict(timages)
    for i in range(len(files)):
        plt.imshow(np.ndarray.astype(pimages[i], np.uint8))
        plt.show()

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")
