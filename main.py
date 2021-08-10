import tensorflow as tf
from load_data import load_book30
import models
from train import train
from plot import plot


if __name__ == "__main__":
    train_ds, test_ds = load_book30(batch_size=64, image_size=(56, 56), genre=24)
    vocab_size = 5000
    dataset_name = "book30-scifi"
    n_epochs = tf.constant(15)
    sample_titles = tf.constant(["Daughter of Darkness", "The Lord of the Rings"])

    # tf.config.run_functions_eagerly(True)

    t2c = models.Title2Cover(train_ds, vocab_size, dataset_name, restore_checkpoint=True)
    train(t2c, train_ds, sample_titles, n_epochs, dataset_name)

    # gen = models.generator(train_ds, vocab_size)
    # plot(gen, sample_titles)
    # disc = models.discriminator(tf.TensorShape([224, 224, 3]))
