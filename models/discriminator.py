import tensorflow as tf
from layers import ConvolutionBlock


def discriminator(image_shape: tf.TensorShape) -> tf.keras.Model:
    return discriminator2(image_shape)


def discriminator1(image_shape: tf.TensorShape) -> tf.keras.Model:
    """Initialize model that discriminates between the generated cover"""
    leaky_slope = 0.2
    inputs = tf.keras.layers.Input(shape=image_shape)
    x = ConvolutionBlock(4, 2, 64, normalize=False, leaky_slope=0.2)(inputs)  # (bs, res/2, res/2, 64)
    x = ConvolutionBlock(4, 2, 128, leaky_slope=leaky_slope)(x)  # (bs, res/4, res/4, 128)
    x = ConvolutionBlock(4, 2, 256, leaky_slope=leaky_slope)(x)  # (bs, res/8, res/8, 256)
    # x = ConvolutionBlock(4, 2, 512, leaky_slope=leaky_slope)(x)  # (bs, res/16, res/16, 512)
    x = ConvolutionBlock(kernel_size=4, stride=1, n_filters=512, normalize=True)(x)
    # Final convolution to produce 1D output (bs, res/16, res/16, 1)
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding="same",
                               kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def discriminator2(image_shape: tf.TensorShape) -> tf.keras.Model:
    """Initialize model that discriminates between the generated cover"""
    ls = 0.2  # leaky slope
    kernel = 4
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape=image_shape)
    x = ConvolutionBlock(3, 1, 32, normalize=True, dropout=False, leaky_slope=ls)(inputs)  # (bs, res, res, 128)
    x = ConvolutionBlock(kernel, 2, 64, normalize=True, dropout=True, leaky_slope=ls)(x)  # (bs, r/2, r/2, 128)
    x = ConvolutionBlock(kernel, 2, 128, normalize=True, dropout=True, leaky_slope=ls)(x)  # (bs, r/4, r/4, 128)
    x = ConvolutionBlock(kernel, 2, 256, normalize=True, dropout=True, leaky_slope=ls)(x)  # (bs, r/8, r/8, 128)
    # Final convolution to produce 1D output (bs, res/8, res/8, 1)
    x = tf.keras.layers.Conv2D(kernel_size=kernel, strides=1, filters=1, padding="same", use_bias=False,
                               kernel_initializer=initializer)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)
