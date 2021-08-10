import tensorflow as tf
from layers import ConvolutionBlock


def generator(train_ds: tf.data.Dataset, vocabulary_size: int) -> tf.keras.Model:
    return generator2(train_ds, vocabulary_size)


def generator1(train_ds: tf.data.Dataset, vocabulary_size: int) -> tf.keras.Model:
    """
    Initialize model that generates a book cover from the title text.

    Parameters
    ----------
    train_ds: tf.data.Dataset
        Training dataset to be used to initialize the vocabulary in the embedding layer.
    vocabulary_size: int
        Vocabulary size.

    Returns
    -------
    tf.keras.Model
    """
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=vocabulary_size)
    encoder.adapt(train_ds.map(return_title))

    ls = 0.2  # leaky relu slope

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1,), dtype=tf.dtypes.string),
        encoder,
        # Vectorizes tokenized words (mask_zero causes padding values to be ignored)
        tf.keras.layers.Embedding(vocabulary_size, 64, mask_zero=True),
        # Return_sequences causes the full sequences of predictions to be returned rather than just the final
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(49, activation="relu"),
        tf.keras.layers.Reshape((7, 7, 1)),
        # Now generate an image. ConvBlock order is kernel size, stride, and n filters
        ConvolutionBlock(3, 2, 128, transpose=True, normalize=True, dropout=True, leaky_slope=ls),  # (bs,14,14,128)
        ConvolutionBlock(3, 2, 64, transpose=True, normalize=True, dropout=True, leaky_slope=ls),  # (bs,28,28,64)
        # ConvolutionBlock(3, 2, 32, transpose=True, normalize=True, dropout=True, leaky_slope=ls),  # (bs,56,56,32)
        # ConvolutionBlock(3, 2, 16, transpose=True, normalize=True, dropout=True, leaky_slope=ls),  # (bs,112,112,16)
        ConvolutionBlock(3, 2, 3, transpose=True, normalize=True, dropout=True, leaky_slope=ls),  # (bs,224,224,3)
    ])
    return model


def generator2(train_ds: tf.data.Dataset, vocabulary_size: int) -> tf.keras.Model:
    """
    Initialize model that generates a book cover from the title text.

    This is a simpler version of generator1.  Trying to get it to train faster.

    Parameters
    ----------
    train_ds: tf.data.Dataset
        Training dataset to be used to initialize the vocabulary in the embedding layer.
    vocabulary_size: int
        Vocabulary size.

    Returns
    -------
    tf.keras.Model
    """
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=vocabulary_size)
    encoder.adapt(train_ds.map(return_title))

    ls = 0.2  # leaky relu slope
    n_embedding = 128
    initializer = tf.random_normal_initializer(0., 0.02)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1,), dtype=tf.dtypes.string),
        encoder,
        # Vectorizes tokenized words (mask_zero causes padding values to be ignored)
        tf.keras.layers.Embedding(vocabulary_size, n_embedding, mask_zero=True),
        # Return_sequences causes the full sequences of predictions to be returned rather than just the final
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_embedding, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_embedding // 2)),
        tf.keras.layers.Dense(7 * 7 * 256, use_bias=False),
        tf.keras.layers.Reshape((7, 7, 256)),
        # Now generate an image. ConvBlock order is kernel size, stride, and n filters
        ConvolutionBlock(5, 1, 128, transpose=False, normalize=True, dropout=False, leaky_slope=ls),  # (bs,7,7,128)
        ConvolutionBlock(2, 2, 64, transpose=True, normalize=True, dropout=False, leaky_slope=ls),  # (bs,14,14,64)
        ConvolutionBlock(2, 2, 32, transpose=True, normalize=True, dropout=False, leaky_slope=ls),  # (bs,28,28,32)
        ConvolutionBlock(2, 2, 16, transpose=True, normalize=True, dropout=False, leaky_slope=ls),  # (bs,56,56,16)
        ConvolutionBlock(5, 1, 8, transpose=False, normalize=True, dropout=False, leaky_slope=ls),  # (bs,56,56,8)
        tf.keras.layers.Conv2D(kernel_size=5, strides=1, filters=3,
                               padding="same", use_bias=False, activation="tanh",
                               kernel_initializer=initializer),  # (bs,224,224,3)
    ])
    return model


def return_title(title: tf.Tensor, cover: tf.Tensor):
    """Returns the title given a book's title and cover."""
    return title
