import os
import tensorflow as tf
import models


class WGAN(tf.keras.Model):

    """WGAN for taking book titles and generating covers."""

    # Optimizer parameters
    gen_learning_rate = 1e-4
    disc_learning_rate = 1e-4
    beta1 = 0.
    beta2 = 0.9

    # Number of times to train the critic for every one time the generator is trained
    n_critic = 5

    gradient_penalty_coefficient = tf.constant(10.)

    def __init__(self,
                 train_ds: tf.data.Dataset,
                 vocabulary_size: int,
                 dataset_name: str,
                 restore_checkpoint: bool = True
                 ):

        super(WGAN, self).__init__()

        """Set loss object and random number generator"""
        self.random = tf.random.Generator.from_non_deterministic_state()

        """Initialize generator and discriminator"""
        # This line causes a warning to come up saying the partially cached data will be discarded.
        batch = next(iter(train_ds))
        self.generator = models.title2cover.generator(train_ds, vocabulary_size)
        self.discriminator = models.title2cover.discriminator(batch[1].shape[1:])

        """"Set up optimizers"""
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.gen_learning_rate,
                                                            beta_1=self.beta1, beta_2=self.beta2)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.disc_learning_rate,
                                                                beta_1=self.beta1, beta_2=self.beta2)

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

    def discriminator_loss(self, disc_real: tf.Tensor, disc_generated: tf.Tensor,
                           gradient_penalty: tf.Tensor) -> tf.Tensor:
        """
        Discriminator wants to get better at detecting that the real image is real, and that generated images are not.
    Softens labels.  Supposed to improve training.
        """
        loss = tf.constant(-1.) * tf.reduce_mean(disc_real) + tf.reduce_mean(disc_generated)
        regularized_loss = loss + gradient_penalty * self.gradient_penalty_coefficient
        return regularized_loss

    @staticmethod
    def generator_loss(disc_generated: tf.Tensor) -> tf.Tensor:
        """Generator wants to get closer to creating a realistic image."""
        return tf.constant(-1.) * tf.reduce_mean(disc_generated)

    @tf.function
    def gradient_penalty(self, images: tf.Tensor, fake_images: tf.Tensor):
        """
        Compute the gradient penalty.  When using the gradient penalty technique:
            Do not use BatchNorm in the discriminator
            Do not use the weight clipping constraint
        """
        epsilion = self.random.uniform(shape=(images.shape[0], 1, 1, 1), minval=0., maxval=1.)
        diff = images - fake_images
        perturbed_images = fake_images + epsilion * diff

        with tf.GradientTape() as tape:
            # This is a tensor so the tape must be told to watch it.  By default it only watches tf.Variables.
            tape.watch(perturbed_images)
            disc_pert_images = self.discriminator(perturbed_images)

        gradients = tape.gradient(disc_pert_images, [perturbed_images])
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        grad_penalty = tf.reduce_mean((norm - 1) ** 2)
        return grad_penalty

    @tf.function
    def train_discriminator(self,
                            titles: tf.Tensor,
                            images: tf.Tensor,
                            summary_writer: tf.summary.SummaryWriter,
                            step: tf.Tensor
                            ):
        with tf.GradientTape(persistent=False) as tape:
            # Generate fake covers
            fake_images = self.generator(titles)
            # Calculate the gradient penalty
            gp = self.gradient_penalty(images, fake_images)
            # Discriminate real covers - are real covers realistic?  (should be!)
            disc_real = self.discriminator(images)
            # Discriminate fake covers - are fake covers realistic?
            disc_fake = self.discriminator(fake_images)
            # How real are real covers?  How fake are fake covers?
            disc_loss = self.discriminator_loss(disc_real, disc_fake, gp)

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
    def train_generator(self,
                        titles: tf.Tensor,
                        images: tf.Tensor,
                        summary_writer: tf.summary.SummaryWriter,
                        step: tf.Tensor
                        ):

        with tf.GradientTape(persistent=False) as tape:
            # Generate fake covers
            fake_images = self.generator(titles)
            # Discriminate fake covers - are fake covers realistic?
            disc_fake = self.discriminator(fake_images)
            # How close are the fake covers to real covers?
            gen_loss = self.generator_loss(disc_fake)

            # Calculate gradients for the generator
            # Ask the optimizers to apply the gradients to the trainable variables
            gen_gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

            with summary_writer.as_default():
                tf.summary.scalar("loss/generator", gen_loss, step=step)
                for i, var in enumerate(self.generator.trainable_variables):
                    tf.summary.scalar(f"gradients/generator/{var.name}", tf.reduce_mean(gen_gradients[i]), step=step)
