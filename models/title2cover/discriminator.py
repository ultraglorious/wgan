import tensorflow as tf
from layers import ConvolutionBlock as ConvBlock
from models.constraints import ClipConstraint


def discriminator(image_shape: tf.TensorShape) -> tf.keras.Model:
    return wgan_critic(image_shape)


def wgan_critic(image_shape: tf.TensorShape) -> tf.keras.Model:
    """Initialize WGAN critic model"""
    ls = 0.2  # leaky slope
    initializer = tf.random_normal_initializer(0., 0.02)

    inputs = tf.keras.layers.Input(shape=image_shape)
    x = ConvBlock(5, 2, 64, normalize=False, dropout=True, leaky_slope=ls)(inputs)  # r/2, r/2, 64
    x = ConvBlock(5, 2, 128, normalize=False, dropout=True, leaky_slope=ls)(x)  # r/4, r/4, 128
    x = ConvBlock(5, 2, 256, normalize=False, dropout=True, leaky_slope=ls)(x)  # r/8, r/8, 256
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1, activation="linear", kernel_initializer=initializer)(x)  # (bs, 1)
    return tf.keras.Model(inputs=inputs, outputs=x)
