from typing import Optional
import tensorflow as tf


class ConvolutionBlock(tf.keras.layers.Layer):

    """Creates composite Conv2D(Transpose)-InstanceNormalization-(Leaky)ReLU layer."""

    def __init__(self, kernel_size: int, stride: int, n_filters: int,
                 transpose: bool = False,
                 normalize: bool = True,
                 dropout: bool = False,
                 activation: str = "relu",
                 leaky_slope: Optional[float] = None,
                 *args, **kwargs):
        """
        Parameters
        ----------
        kernel_size: int
            Kernel size.
        stride: int
            Stride size.
        n_filters: int
            Number of filters.
        transpose: bool
            Set to True to do a transpose (upscaling) convolution instead.
            This should be what 'fractional stride' means.
        normalize: bool
            If set to True, normalize using InstanceNormalization.
        dropout: bool
            If True, apply dropout layer.
        activation: str = "relu"
            String name of activation type.  Defaults to "relu".
        leaky_slope: float = None
            Slope of LeakyReLU layer.  This is the slope for x < 0.  Defaults to None.
        """
        super(ConvolutionBlock, self).__init__(*args, **kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)

        if transpose:
            self.conv = tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=kernel_size, strides=stride,
                                                        padding="same", kernel_initializer=initializer, use_bias=False)
        else:
            self.conv = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size, strides=stride,
                                               padding="same", kernel_initializer=initializer, use_bias=False)

        self.norm = None
        if normalize:
            self.norm = tf.keras.layers.BatchNormalization()

        self.dropout = None
        if dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)

        if (activation == "relu") and (leaky_slope is not None):
            self.acti = tf.keras.layers.LeakyReLU(alpha=leaky_slope)
        else:
            self.acti = tf.keras.layers.Activation(activation=activation)

    def call(self, inputs: tf.Tensor):
        x = inputs
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.acti(x)
