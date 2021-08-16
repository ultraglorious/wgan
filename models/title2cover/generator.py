import tensorflow as tf
from layers import ConvolutionBlock as ConvBlock


def generator(train_ds: tf.data.Dataset, vocabulary_size: int) -> tf.keras.Model:
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
    n_embedding = 16
    initializer = tf.random_normal_initializer(0., 0.02)

    inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.dtypes.string)
    x = encoder(inputs)
    # Vectorizes tokenized words (mask_zero causes padding values to be ignored)
    x = tf.keras.layers.Embedding(vocabulary_size, n_embedding, mask_zero=True)(x)

    # Return_sequences causes the full sequences of predictions to be returned rather than just the final
    # x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_embedding, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_embedding))(x)

    x = tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, kernel_initializer=initializer)(x)
    x = tf.keras.layers.Reshape((7, 7, 256))(x)

    x = ConvBlock(5, 1, 256, transpose=True, normalize=True, dropout=True, leaky_slope=ls)(x)  # 7, 7, 256
    x = ConvBlock(5, 2, 128, transpose=True, normalize=True, dropout=True, leaky_slope=ls)(x)  # 14, 14, 128
    x = ConvBlock(5, 2, 64, transpose=True, normalize=True, dropout=True, leaky_slope=ls)(x)  # 25, 28, 64
    x = ConvBlock(5, 2, 3, transpose=True, normalize=True, dropout=True, activation="tanh")(x)  # 56, 56, 3

    return tf.keras.Model(inputs=inputs, outputs=x)


def return_title(title: tf.Tensor, cover: tf.Tensor):
    """Returns the title given a book's title and cover."""
    return title
