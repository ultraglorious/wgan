import tensorflow as tf
import models
import load

from plot import plot


def run_title2cover():
    from train_t2c import train

    train_ds, test_ds = load.book30.load(batch_size=128, image_size=(56, 56), genre=24)
    vocab_size = 1000
    dataset_name = "book30-scifi"
    n_epochs = tf.constant(50)
    sample_titles = tf.constant(["Daughter of Darkness", "The Lord of the Rings", "Pushing Ice"])

    # tf.config.run_functions_eagerly(True)

    t2c = models.title2cover.Title2Cover(train_ds, vocab_size, dataset_name, restore_checkpoint=True)
    train(t2c, train_ds, sample_titles, n_epochs, dataset_name)


def run_mnist():
    from train_digits import train

    batch_size = 128
    train_ds = load.mnist.load(batch_size=batch_size)
    dataset_name = "mnist-digits"
    rand = tf.random.Generator.from_seed(0)
    n_epochs = tf.constant(10)

    wgan = models.mnist.WGAN(dataset_name=dataset_name)
    sample = rand.normal(shape=(3, wgan.latent_dimensions))
    train(wgan, train_ds, sample, n_epochs, dataset_name)


if __name__ == "__main__":
    run_mnist()
