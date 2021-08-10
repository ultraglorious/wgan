import os
import datetime
import tensorflow as tf
from models import Title2Cover
from plot import plot


def train(model: Title2Cover,
          dataset: tf.data.Dataset,
          sample_titles: tf.Tensor,
          n_epochs: tf.Tensor,
          dataset_name: str):

    # Restore latest checkpoint
    start_epoch = tf.constant(0)
    if model.checkpoint_manager.latest_checkpoint:
        start_epoch = tf.cast(model.checkpoint.save_counter, dtype=tf.int32) * model.save_every_nth
        model.checkpoint.restore(model.checkpoint_manager.latest_checkpoint)

    log_dir = os.path.join(os.getcwd(), "data", "log")
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, dataset_name, time))

    for epoch in tf.range(start_epoch, n_epochs):
        start = tf.timestamp()

        n = 0
        for titles, images in dataset:
            model.train_step(titles, images, summary_writer, epoch)
            if tf.math.equal(tf.math.floormod(n, 10), 0):
                tf.print(".", end="")
            n += 1

        # Using a consistent image (sample_x) so that the progress of the model is clearly visible.
        directory = os.path.join(os.getcwd(), "data", "output", dataset_name)
        plot(model.generator, sample_titles, epoch, directory)

        if tf.equal(tf.math.floormod(epoch + 1, model.save_every_nth), 0):
            save_path = model.checkpoint_manager.save()
            tf.print(f"Saving checkpoint for epoch {epoch + 1} at {save_path}")
        running_time = tf.timestamp() - start
        tf.print(f"Time taken for epoch {epoch + 1} is {running_time:.1f} sec\n")
