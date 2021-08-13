import tensorflow as tf
from layers import ConvolutionBlock as ConvBlock
from models.constraints import ClipConstraint


def discriminator() -> tf.keras.Model:
    """Initialize WGAN critic model"""
    ls = 0.2  # leaky slope
    initializer = tf.random_normal_initializer(0., 0.02)

    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.GaussianNoise(stddev=0.2)(inputs)
    x = ConvBlock(5, 2, 64, normalize=False, dropout=True, leaky_slope=ls)(x)  # r/2, r/2, 64
    x = ConvBlock(5, 2, 128, normalize=False, dropout=True, leaky_slope=ls)(x)  # r/4, r/4, 128
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1, activation="linear", kernel_initializer=initializer)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)
