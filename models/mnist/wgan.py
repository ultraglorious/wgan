import os
import tensorflow as tf
import models


class WGAN(tf.keras.Model):

    """WGAN for taking book titles and generating covers."""

    # Optimizer parameters
    gen_learning_rate = 5e-5
    disc_learning_rate = 5e-5
    # Number of times to train the critic for every one time the generator is trained
    n_critic = 5
    latent_dimensions = 100

    def __init__(self,
                 dataset_name: str,
                 restore_checkpoint: bool = True
                 ):

        super(WGAN, self).__init__()

        """Set loss object and random number generator"""
        self.random = tf.random.Generator.from_non_deterministic_state()

        """Initialize generator and discriminator"""
        # This line causes a warning to come up saying the partially cached data will be discarded.
        self.generator = models.mnist.generator(latent_dim=self.latent_dimensions)
        self.discriminator = models.mnist.discriminator()

        """"Set up optimizers"""
        self.generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.gen_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.disc_learning_rate)

        """Set up checkpointing"""
        self.checkpointing(dataset_name)
        if restore_checkpoint:
            self.restore_checkpoint()

    def checkpointing(self, dataset_name: str):
        """Set up checkpointing"""
        self.save_every_nth = tf.constant(1)  # Save every nth checkpoint
        checkpoint_path = os.path.join(os.getcwd(), "data", "checkpoints", dataset_name, "train")
        self.checkpoint = tf.train.Checkpoint(generator=self.generator,
                                              discriminator=self.discriminator,
                                              generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_path, max_to_keep=5)

    def restore_checkpoint(self):
        """If a checkpoint exists, restore the latest checkpoint."""
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print("Latest checkpoint restored.")

    @staticmethod
    def discriminator_loss(disc_real: tf.Tensor, disc_generated: tf.Tensor) -> tf.Tensor:
        """
        Discriminator wants to get better at detecting that the real image is real, and that generated images are not.
        Softens labels.  Supposed to improve training.
        """
        return tf.constant(-1.) * tf.reduce_mean(disc_real) + tf.reduce_mean(disc_generated)

    @staticmethod
    def generator_loss(disc_generated: tf.Tensor) -> tf.Tensor:
        """Generator wants to get closer to creating a realistic image."""
        return tf.constant(-1.) * tf.reduce_mean(disc_generated)

    @tf.function
    def train_only_discriminator(self,
                                 images: tf.Tensor,
                                 summary_writer: tf.summary.SummaryWriter,
                                 step: tf.Tensor
                                 ):

        noise_tensor = self.random.normal(shape=(images.shape[0], self.latent_dimensions))

        with tf.GradientTape(persistent=True) as tape:
            # Generate fake covers
            fake_images = self.generator(noise_tensor)
            # Discriminate real covers - are real covers realistic?  (should be!)
            disc_real = self.discriminator(images)
            # Discriminate fake covers - are fake covers realistic?
            disc_fake = self.discriminator(fake_images)
            # How real are real covers?  How fake are fake covers?
            disc_loss = self.discriminator_loss(disc_real, disc_fake)

        # Calculate gradients for just the discriminator
        disc_gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        # Ask the optimizers to apply the gradients to the trainable variables
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar("loss/discriminator", disc_loss, step=step)
            disc_acc_real = tf.keras.metrics.binary_accuracy(tf.ones_like(disc_real), disc_real, threshold=0.5)
            disc_acc_fake = tf.keras.metrics.binary_accuracy(tf.zeros_like(disc_fake), disc_fake, threshold=0.5)
            tf.summary.scalar("discriminator accuracy/real", tf.reduce_mean(disc_acc_real), step=step)
            tf.summary.scalar("discriminator accuracy/fake", tf.reduce_mean(disc_acc_fake), step=step)
            for i, var in enumerate(self.discriminator.trainable_variables):
                tf.summary.scalar(f"gradients/discriminator/{var.name}", tf.reduce_mean(disc_gradients[i]), step=step)

    @tf.function
    def train_step(self,
                   images: tf.Tensor,
                   summary_writer: tf.summary.SummaryWriter,
                   step: tf.Tensor
                   ):

        noise_tensor = self.random.normal(shape=(images.shape[0], self.latent_dimensions))

        with tf.GradientTape(persistent=True) as tape:
            # Generate fake covers
            fake_images = self.generator(noise_tensor)
            # Discriminate real covers - are real covers realistic?  (should be!)
            disc_real = self.discriminator(images)
            # Discriminate fake covers - are fake covers realistic?
            disc_fake = self.discriminator(fake_images)
            # How close are the fake covers to real covers?
            gen_loss = self.generator_loss(disc_fake)
            # How real are real covers?  How fake are fake covers?
            disc_loss = self.discriminator_loss(disc_real, disc_fake)

        # Calculate gradients for the generator and discriminator
        gen_gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        # Ask the optimizers to apply the gradients to the trainable variables
        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar("loss/generator", gen_loss, step=step)
            tf.summary.scalar("loss/discriminator", disc_loss, step=step)
            disc_acc_real = tf.keras.metrics.binary_accuracy(tf.ones_like(disc_real), disc_real, threshold=0.5)
            disc_acc_fake = tf.keras.metrics.binary_accuracy(tf.zeros_like(disc_fake), disc_fake, threshold=0.5)
            tf.summary.scalar("discriminator accuracy/real", tf.reduce_mean(disc_acc_real), step=step)
            tf.summary.scalar("discriminator accuracy/fake", tf.reduce_mean(disc_acc_fake), step=step)
            for i, var in enumerate(self.generator.trainable_variables):
                tf.summary.scalar(f"gradients/generator/{var.name}", tf.reduce_mean(gen_gradients[i]), step=step)
            for i, var in enumerate(self.discriminator.trainable_variables):
                tf.summary.scalar(f"gradients/discriminator/{var.name}", tf.reduce_mean(disc_gradients[i]), step=step)
