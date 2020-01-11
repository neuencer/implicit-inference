from math import ceil

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class JCAdersarialVI:
    """Implements joint-constrastive adversarial variational inference from
    https://arxiv.org/pdf/1702.08235.pdf
    
    Args:
        generator: Function. The generator
        discriminator: tf.keras.Model. The discriminator model
        recognition: tf.keras.Model. The recognition model
        learning_rate: Float. The learning rate to use
    """
    def __init__(self, generator, discriminator, recognition, prior,
                 learning_rate=2e-4):
        self.generator = generator
        self.discriminator = discriminator
        self.recognition = recognition
        self.prior = prior
        noise_dim = recognition.input_shape[1][1]
        self.noise = tfd.Normal(loc=tf.zeros([noise_dim]),
                                scale=tf.ones([noise_dim]))
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate,
                                                                beta_1=0.5)
        self.recognition_optimizer = tf.keras.optimizers.Adam(learning_rate,
                                                              beta_1=0.5)
    
    @tf.function
    def update(self, inputs, update_recognition=True):
        """Implements one step of the joint-contrastive adversarial
        algorithm.
        
        Args:
            inputs: tf.Tensor. The input data
            update_recognition: Boolean. Wether to update the recognition model
        
        Returns:
            tf.Tensor, tf.Tensor: discriminator loss, recognition loss
        """
        batch_size = tf.shape(inputs)[0]
        # persistent is set to True because the tape is used more than once
        # to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:

            # Inputs --> latent variables
            noise = self.noise.sample(batch_size)
            z_q = self.recognition([inputs, noise])

            # Prior sample --> generated data
            z_p = self.prior.sample(batch_size)
            x_p = self.generator(z_p)

            # Density ratio estimation
            log_q = self.discriminator([inputs, z_q])
            log_p = self.discriminator([x_p, z_p])

            # Calculate discriminator loss
            discriminator_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(log_p),
                                                        log_p) + \
                tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(log_q),
                                                        log_q))
            
            # Calculate recognition loss
            recognition_loss = -tf.reduce_mean(log_q)
        
        # Calculate the gradients and apply to optimizer
        discriminator_gradients = tape.gradient(
            discriminator_loss,
            self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients,
                self.discriminator.trainable_variables))

        if update_recognition:
            recognition_gradients = tape.gradient(
                recognition_loss,
                self.recognition.trainable_variables)
            self.recognition_optimizer.apply_gradients(
                zip(recognition_gradients,
                    self.recognition.trainable_variables))
        
        return discriminator_loss, recognition_loss

    def fit(self, dataset, steps_per_epoch=1, discriminator_steps=1):
        """Trains the recognition and discriminator model for a full iteration
        through the dataset.
        
        Args:
            dataset: tf.Dataset: Should return the data
            steps_per_epoch: Int. Total number of steps before declaring one
            epoch finished and starting the next epoch.
            discriminator_steps: Total number of discriminator updates before
            a full training step.
        """
        cardinallity = tf.data.experimental.cardinality(dataset).numpy()
        nb_epochs = ceil(cardinallity / (steps_per_epoch * discriminator_steps))
        epoch = 0
        step = 0
        for x in dataset:
            # Initialize progressbar for new epoch
            # An epoch consists of steps_per_epoch * discriminator_steps
            # updates of the discriminator and steps_per_epoch updates of the
            # recognition model
            if step % (discriminator_steps * steps_per_epoch) == 0:
                epoch += 1
                print('\nEpoch {} of {}'.format(epoch, nb_epochs))
                progbar = tf.keras.utils.Progbar(target=steps_per_epoch)
            
            # Only train recognition model on every n-th step, where n is the
            # number of discriminator steps
            if step % discriminator_steps != 0:
                self.update(x, update_recognition=False)
            else:
                discriminator_loss, recognition_loss = self.update(x)
                progbar.update(
                    current=step // discriminator_steps % steps_per_epoch,
                    values=[('cross_entropy', discriminator_loss),
                            ('-log(q)', recognition_loss)]
                )
            step += 1
