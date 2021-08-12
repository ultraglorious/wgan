import tensorflow as tf
from layers import ConvolutionBlock as ConvBlock
from models.constraints import ClipConstraint


def discriminator() -> tf.keras.Model:
    return wgan_critic()


def wgan_critic() -> tf.keras.Model:
    """Initialize WGAN critic model"""
    ls = 0.2  # leaky slope
    initializer = tf.random_normal_initializer(0., 0.02)
    cv = 0.01  # clip value
    clip = ClipConstraint(cv)

    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = ConvBlock(5, 2, 128, normalize=True, dropout=True, leaky_slope=ls, clip_value=cv)(inputs)  # r/2, r/2, 256
    x = ConvBlock(5, 2, 128, normalize=True, dropout=True, leaky_slope=ls, clip_value=cv)(x)  # r/4, r/4, 256
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1, activation="linear", kernel_initializer=initializer, kernel_constraint=clip)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)
