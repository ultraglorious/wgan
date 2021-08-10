import os
import tensorflow as tf
import models


class Title2Cover(tf.keras.Model):

    """Model for taking book titles and generating covers."""

    # Optimizer parameters
    gen_learning_rate = 2e-4
    disc_learning_rate = 2e-4
    beta1 = 0.5  # The exponential decay rate for the 1st moment estimates. (Not sure what this is really)

    def __init__(self,
                 train_ds: tf.data.Dataset,
                 vocabulary_size: int,
                 dataset_name: str,
                 restore_checkpoint: bool = True
                 ):

        super(Title2Cover, self).__init__()

        """Set loss object and random number generator"""
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.loss_penalty = 0.5
        self.random = tf.random.Generator.from_non_deterministic_state()

        """Initialize generator and discriminator"""
        # This line causes a warning to come up saying the partially cached data will be discarded.
        batch = next(iter(train_ds))
        self.generator = models.generator(train_ds, vocabulary_size)
        self.discriminator = models.discriminator(batch[1].shape[1:])

        """"Set up optimizers"""
        self.generator_optimizer = tf.keras.optimizers.Adam(self.gen_learning_rate, beta_1=self.beta1)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.disc_learning_rate, beta_1=self.beta1)

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

    def soft_labels(self, image: tf.Tensor, ones: bool = True):
        """
        Create a tensor of soft labels (1 for real and 0 for fake) by adding or subtracting a random small component.
        """
        rand_shape = tf.constant([1])
        # Are the labels for this image ones or zeros (real or fake)
        if ones:
            like_tensor = tf.ones_like(image)
            soft_direction = tf.constant([-1.])
        else:
            like_tensor = tf.zeros_like(image)
            soft_direction = tf.constant([1.])

        # For ones_like, the values end up between 0.9 and 1.  For zeros_like, it is between 0 and 0.1.
        return like_tensor + soft_direction * self.random.uniform(rand_shape, minval=0, maxval=0.1)

    def discriminator_loss(self, real_image: tf.Tensor, generated_image: tf.Tensor) -> tf.Tensor:
        """
        Discriminator wants to get better at detecting that the real image is real, and that generated images are not.
        Softens labels.  Supposed to improve training.
        TODO: set a 5% chance of flipping labels
        """
        if tf.math.less_equal(self.random.uniform([1]), 0.05):
            # 5% chance of flipping labels
            # This means: real images are fake and fake images are real
            real_loss = self.loss_object(self.soft_labels(image=generated_image, ones=True), generated_image)
            gen_loss = self.loss_object(self.soft_labels(image=real_image, ones=False), real_image)
        else:
            # Real images are real and generated images are not
            real_loss = self.loss_object(self.soft_labels(image=real_image, ones=True), real_image)
            gen_loss = self.loss_object(self.soft_labels(image=generated_image, ones=False), generated_image)
        return (real_loss + gen_loss) * self.loss_penalty

    def generator_loss(self, generated_image: tf.Tensor) -> tf.Tensor:
        """Generator wants to get closer to creating a realistic image."""
        return self.loss_object(tf.ones_like(generated_image), generated_image)

    @tf.function
    def train_step(self,
                   titles: tf.Tensor,
                   images: tf.Tensor,
                   summary_writer: tf.summary.SummaryWriter,
                   epoch: tf.Tensor
                   ):
        with tf.GradientTape(persistent=True) as tape:
            # Generate fake covers
            fake_images = self.generator(titles)
            # Discriminate real covers - are real covers realistic?  (should be!)
            disc_real = self.discriminator(images)
            # Discriminate fake covers - are fake covers realistic?
            disc_fake = self.discriminator(fake_images)
            # How close are the fake covers to real covers?
            gen_loss = self.generator_loss(disc_fake)
            # How real are real covers?  How fake are fake covers?
            disc_loss = self.discriminator_loss(disc_real, disc_fake)
            # Loss function for testing (comment out above)
            # gen_loss = self.loss_object(images, fake_images)

        # Calculate gradients for the generator and discriminator
        gen_gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        # Ask the optimizers to apply the gradients to the trainable variables
        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        with summary_writer.as_default():
            step = tf.cast(epoch, tf.dtypes.int64)
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

