import os
import numpy as np
import tensorflow as tf


def reshape_and_normalize(images: np.ndarray) -> np.ndarray:
    """Adds a channel dimension and normalizes images between -1 and 1"""
    images = images.reshape((images.shape[0], 28, 28, 1)).astype('float32')
    return (images - 127.5) / 127.5


def make_dataset(images: np.ndarray, labels: np.ndarray, batch_size: int):
    """Make dataset of MNIST digits"""
    buffer_size = 1000
    cache_fn = os.path.join(os.getcwd(), "../data", "cache")

    dataset = tf.data.Dataset.from_tensor_slices((labels, reshape_and_normalize(images)))
    dataset = dataset.cache().shuffle(buffer_size).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def load(batch_size: int):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_ds = make_dataset(train_images, train_labels, batch_size)
    return train_ds
