import tensorflow as tf
from layers import ConvolutionBlock as ConvBlock
from models.constraints import ClipConstraint


def discriminator(image_shape: tf.TensorShape) -> tf.keras.Model:
    return wgan_critic(image_shape)


def wgan_critic(image_shape: tf.TensorShape) -> tf.keras.Model:
    """Initialize WGAN critic model"""
    ls = 0.2  # leaky slope
    kernel = 4
    initializer = tf.random_normal_initializer(0., 0.02)
    cv = 0.01  # clip value
    clip = ClipConstraint(cv)

    inputs = tf.keras.layers.Input(shape=image_shape)
    x = ConvBlock(kernel, 2, 128, normalize=True, dropout=True, leaky_slope=ls, clip_value=cv)(inputs)  # r/2, r/2, 256
    x = ConvBlock(kernel, 2, 128, normalize=True, dropout=True, leaky_slope=ls, clip_value=cv)(x)  # r/4, r/4, 256
    x = ConvBlock(kernel, 2, 128, normalize=True, dropout=True, leaky_slope=ls, clip_value=cv)(x)  # r/8, r/8, 256
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1, activation="linear", kernel_initializer=initializer, kernel_constraint=clip)(x)  # (bs, 1)
    return tf.keras.Model(inputs=inputs, outputs=x)
