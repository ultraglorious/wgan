import os
from typing import Optional
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def show_tensor(axis: plt.Axes, t: tf.Tensor):
    axis.imshow((t - tf.reduce_min(t)) / (tf.reduce_max(t) - tf.reduce_min(t)))


def plot(generator: tf.keras.Model, test_input: tf.Tensor,
         epoch: Optional[int] = None, directory: Optional[str] = None):

    if epoch is not None:
        backend = "Agg"
    else:
        backend = "TkAgg"
    if mpl.get_backend() != backend:
        mpl.use(backend)  # Do not create a plotting window but plot to a buffer

    if test_input.shape.rank == 0:
        test_input = tf.expand_dims(test_input, axis=0)
        n_images = 1
    else:
        n_images = test_input.shape[0]

    filename = "image_at_epoch_{:04d}.png"
    fig = plt.figure(figsize=(4 * n_images, 4))
    gs = mpl.gridspec.GridSpec(1, n_images, figure=fig)

    for i in tf.range(n_images):
        ax1 = fig.add_subplot(gs[0, i])
        show_tensor(ax1, generator(tf.expand_dims(test_input[i], axis=0))[0])
        ax1.set_title(test_input[i].numpy())
        ax1.axis("off")

    if epoch is not None:
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
            fp = os.path.join(directory, filename)
        else:
            fp = filename
        plt.savefig(fp.format(epoch))
    else:
        plt.show()
