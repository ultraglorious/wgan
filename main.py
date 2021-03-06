import tensorflow as tf
import models
import load
import train


def run_title2cover():

    train_ds, test_ds = load.book30.load(batch_size=128, image_size=(56, 56), genre=None)
    vocab_size = 2000  # was using 1000 for scifi (genre=24)
    dataset_name = "book30"
    n_epochs = tf.constant(200)
    sample_titles = tf.constant(["Daughter of Darkness", "The Lord of the Rings", "Pushing Ice"])

    # tf.config.run_functions_eagerly(True)

    t2c = models.title2cover.WGAN(train_ds, vocab_size, dataset_name, restore_checkpoint=True)
    train.t2c(t2c, train_ds, sample_titles, n_epochs, dataset_name)


def run_mnist():
    batch_size = 128
    train_ds = load.mnist.load(batch_size=batch_size)
    dataset_name = "mnist-digits"
    rand = tf.random.Generator.from_seed(0)
    n_epochs = tf.constant(20)

    wgan = models.mnist.WGAN(dataset_name=dataset_name)
    sample = rand.normal(shape=(5, wgan.latent_dimensions))
    train.digits(wgan, train_ds, sample, n_epochs, dataset_name)


if __name__ == "__main__":
    run_title2cover()
