import os
import datetime
import tensorflow as tf
from models.mnist import WGAN
from plot import plot


def train(model: WGAN,
          dataset: tf.data.Dataset,
          sample: tf.Tensor,
          n_epochs: tf.Tensor,
          dataset_name: str):

    # Restore latest checkpoint
    start_epoch = tf.constant(1)
    if model.checkpoint_manager.latest_checkpoint:
        start_epoch = tf.cast(model.checkpoint.save_counter, dtype=tf.int32) * model.save_every_nth
        model.checkpoint.restore(model.checkpoint_manager.latest_checkpoint)

    out_dir = os.path.join(os.getcwd(), "data", "output", dataset_name)
    log_dir = os.path.join(os.getcwd(), "data", "log")
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, dataset_name, time))

    # Untrained images
    plot(model.generator, sample, 0, out_dir)

    step = tf.constant(1, dtype=tf.dtypes.int64)

    for epoch in tf.range(start_epoch, n_epochs+tf.constant(1)):
        start = tf.timestamp()

        for titles, images in dataset:
            if tf.equal(tf.math.floormod(step, model.n_critic), 0):
                # Every fifth batch, train both generator and discriminator/critic
                model.train_generator(images, summary_writer, step)
            # Otherwise, train only the discriminator/critic
            model.train_discriminator(images, summary_writer, step)

            if tf.math.equal(tf.math.floormod(step, 10), 0):
                tf.print(".", end="")
            step += tf.constant(1, dtype=tf.dtypes.int64)

        # Using a consistent image (sample_x) so that the progress of the model is clearly visible.
        plot(model.generator, sample, epoch, out_dir)

        if tf.equal(tf.math.floormod(epoch, model.save_every_nth), 0):
            save_path = model.checkpoint_manager.save()
            tf.print(f"Saving checkpoint for epoch {epoch} at {save_path}")

        running_time = tf.timestamp() - start
        tf.print(f"Time taken for epoch {epoch} is {running_time:.1f} sec\n")
