import tensorflow as tf
import models
import load
from train import train


def run_title2cover():
    train_ds, test_ds = load.book30.load(batch_size=128, image_size=(56, 56), genre=24)
    vocab_size = 1000
    dataset_name = "book30-scifi"
    n_epochs = tf.constant(50)
    sample_titles = tf.constant(["Daughter of Darkness", "The Lord of the Rings", "Pushing Ice"])

    # tf.config.run_functions_eagerly(True)

    t2c = models.title2cover.Title2Cover(train_ds, vocab_size, dataset_name, restore_checkpoint=True)
    train(t2c, train_ds, sample_titles, n_epochs, dataset_name)


def run_mnist():
    train_ds = load.mnist.load(batch_size=128)
    dataset_name = "mnist-digits"


if __name__ == "__main__":
    run_mnist()
