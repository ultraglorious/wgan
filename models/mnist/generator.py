import tensorflow as tf
from layers import ConvolutionBlock as ConvBlock


def generator(latent_dim: int) -> tf.keras.Model:
    """Initializes WGAN generator"""

    ls = 0.2  # leaky relu slope
    initializer = tf.random_normal_initializer(0., 0.02)

    inputs = tf.keras.layers.Input(shape=(latent_dim,), dtype=tf.dtypes.float32)
    x = tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, kernel_initializer=initializer)(inputs)
    x = tf.keras.layers.Reshape((7, 7, 256))(x)
    x = ConvBlock(5, 1, 128, transpose=True, normalize=True, dropout=True, leaky_slope=ls)(x)  # 7, 7, 128
    x = ConvBlock(5, 2, 64, transpose=True, normalize=True, dropout=True, leaky_slope=ls)(x)  # 14, 14, 64
    x = ConvBlock(5, 2, 1, transpose=True, normalize=True, dropout=True, activation="tanh")(x)  # 28, 28, 1

    return tf.keras.Model(inputs=inputs, outputs=x)
