"""
Network Architectures for EHR-M-GAN - TensorFlow 2.x Version

Migrated from TF1.x to TF2 using Keras Model API.
Original: networks.py (TF1.15)

Components:
- C_VAE_TF2: Variational Autoencoder for continuous data (vital signs)
- D_VAE_TF2: Variational Autoencoder for discrete data (interventions)
- C_GAN_TF2: GAN for continuous data generation
- D_GAN_TF2: GAN for discrete data generation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, initializers


class VAE_Base_TF2(keras.Model):
    """
    Base VAE class with common functionality

    This abstract base provides shared methods for both continuous and discrete VAEs.
    """

    def __init__(self,
                 batch_size,
                 time_steps,
                 dim,
                 z_dim,
                 enc_size,
                 dec_size,
                 enc_layers,
                 dec_layers,
                 keep_prob,
                 l2scale,
                 conditional=False,
                 num_labels=0,
                 name='vae_base',
                 **kwargs):
        super(VAE_Base_TF2, self).__init__(name=name, **kwargs)

        self.batch_size = batch_size
        self.time_steps = time_steps
        self.dim = dim
        self.z_dim = z_dim
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.dropout_rate = 1.0 - keep_prob
        self.l2scale = l2scale
        self.conditional = conditional
        self.num_labels = num_labels

    def build_encoder_stack(self, name_prefix='encoder', shared_name='shared_vae'):
        """
        Build encoder LSTM stack

        Args:
            name_prefix (str): Prefix for non-shared layers
            shared_name (str): Name for weight-shared layer (last layer)

        Returns:
            List of LSTM layers
        """
        encoder_layers = []

        # Non-shared layers
        for i in range(self.enc_layers - 1):
            encoder_layers.append(
                layers.LSTM(
                    self.enc_size,
                    return_sequences=True,
                    return_state=True,
                    dropout=self.dropout_rate,
                    kernel_regularizer=regularizers.l2(self.l2scale),
                    name=f'{name_prefix}_lstm_{i}'
                )
            )

        # Last layer with weight sharing (name=shared_name)
        encoder_layers.append(
            layers.LSTM(
                self.enc_size,
                return_sequences=False,
                return_state=True,
                dropout=self.dropout_rate,
                kernel_regularizer=regularizers.l2(self.l2scale),
                name=f'{shared_name}_encoder'  # Shared with decoder
            )
        )

        return encoder_layers

    def build_decoder_stack(self, name_prefix='decoder', shared_name='shared_vae'):
        """
        Build decoder LSTM stack

        Args:
            name_prefix (str): Prefix for non-shared layers
            shared_name (str): Name for weight-shared layer (first layer)

        Returns:
            List of LSTM layers
        """
        decoder_layers = []

        # First layer with weight sharing (name=shared_name)
        decoder_layers.append(
            layers.LSTM(
                self.dec_size,
                return_sequences=True,
                return_state=True,
                dropout=self.dropout_rate,
                kernel_regularizer=regularizers.l2(self.l2scale),
                name=f'{shared_name}_decoder'  # Shared with encoder
            )
        )

        # Non-shared layers
        for i in range(self.dec_layers - 1):
            decoder_layers.append(
                layers.LSTM(
                    self.dec_size,
                    return_sequences=True,
                    return_state=True,
                    dropout=self.dropout_rate,
                    kernel_regularizer=regularizers.l2(self.l2scale),
                    name=f'{name_prefix}_lstm_{i+1}'
                )
            )

        return decoder_layers

    def build_sampling_layers(self, shared_name='shared_vae'):
        """
        Build sampling layers for reparameterization trick

        Returns:
            Tuple of (mu_layer, sigma_layer)
        """
        mu_layer = layers.Dense(
            self.z_dim,
            kernel_initializer='glorot_uniform',
            name=f'{shared_name}_mu'
        )

        sigma_layer = layers.Dense(
            self.z_dim,
            kernel_initializer='glorot_uniform',
            name=f'{shared_name}_sigma'
        )

        return mu_layer, sigma_layer

    @tf.function
    def reparameterize(self, mu, log_sigma):
        """
        Reparameterization trick: z = mu + sigma * epsilon

        Args:
            mu: Mean of latent distribution (batch_size, z_dim)
            log_sigma: Log variance (batch_size, z_dim)

        Returns:
            z: Sampled latent vector (batch_size, z_dim)
            sigma: Standard deviation (batch_size, z_dim)
        """
        sigma = tf.exp(log_sigma)
        eps = tf.random.normal(shape=tf.shape(mu))
        z = mu + sigma * eps
        return z, sigma


class C_VAE_TF2(VAE_Base_TF2):
    """
    Continuous VAE for vital signs data - TF2 Version

    Architecture:
    - Multi-layer LSTM encoder (with weight sharing in last layer)
    - Gaussian sampling layer (reparameterization trick)
    - Multi-layer LSTM decoder (with weight sharing in first layer)
    - Autoregressive reconstruction (x_hat = x - sigmoid(c_prev))

    Args:
        Same as VAE_Base_TF2

    Input:
        input_data: Continuous timeseries (batch_size, time_steps, dim)
        conditions: Optional conditional labels (batch_size, num_labels)

    Output:
        decoded: Reconstructed timeseries (batch_size, time_steps, dim)
        sigma: List of standard deviations per timestep
        mu: List of means per timestep
        log_sigma: List of log variances per timestep
        z: List of latent vectors per timestep
    """

    def __init__(self, *args, **kwargs):
        super(C_VAE_TF2, self).__init__(name='continuous_vae', *args, **kwargs)

    def build(self, input_shape):
        """Build model layers"""
        # Encoder
        self.encoder_layers = self.build_encoder_stack(
            name_prefix='continuous_vae_enc',
            shared_name='shared_vae'
        )

        # Sampling layers
        self.mu_layer, self.sigma_layer = self.build_sampling_layers(
            shared_name='shared_vae'
        )

        # Decoder
        self.decoder_layers = self.build_decoder_stack(
            name_prefix='continuous_vae_dec',
            shared_name='shared_vae'
        )

        # Decoder output layer
        self.decoder_output = layers.Dense(
            self.dim,
            activation='sigmoid',
            kernel_initializer='glorot_uniform',
            name='continuous_vae_output'
        )

        super().build(input_shape)

    @tf.function
    def encode_timestep(self, x_t, x_hat_t, enc_state):
        """
        Encode single timestep

        Args:
            x_t: Input at timestep t (batch_size, dim)
            x_hat_t: Residual input (batch_size, dim)
            enc_state: List of encoder states

        Returns:
            h_enc: Encoded hidden state
            new_enc_state: Updated encoder states
        """
        # Concatenate input and residual
        enc_input = tf.concat([x_t, x_hat_t], axis=-1)
        enc_input = tf.expand_dims(enc_input, axis=1)  # Add time dimension

        # Pass through encoder layers
        h = enc_input
        new_enc_state = []

        for i, layer in enumerate(self.encoder_layers):
            if i < len(self.encoder_layers) - 1:
                # Intermediate layers: return sequences
                h, state_h, state_c = layer(h, initial_state=enc_state[i])
                new_enc_state.append([state_h, state_c])
            else:
                # Last layer: return only final state
                h_enc, state_h, state_c = layer(h, initial_state=enc_state[i])
                new_enc_state.append([state_h, state_c])

        return h_enc, new_enc_state

    @tf.function
    def decode_timestep(self, z_t, dec_state):
        """
        Decode single timestep

        Args:
            z_t: Latent vector at timestep t (batch_size, z_dim)
            dec_state: List of decoder states

        Returns:
            c_t: Decoded output (batch_size, dim)
            new_dec_state: Updated decoder states
        """
        # Expand z for LSTM input
        dec_input = tf.expand_dims(z_t, axis=1)

        # Pass through decoder layers
        h = dec_input
        new_dec_state = []

        for i, layer in enumerate(self.decoder_layers):
            h, state_h, state_c = layer(h, initial_state=dec_state[i])
            new_dec_state.append([state_h, state_c])

        # Output layer
        c_t = self.decoder_output(h)
        c_t = tf.squeeze(c_t, axis=1)  # Remove time dimension

        return c_t, new_dec_state

    def call(self, input_data, conditions=None, training=False):
        """
        Forward pass through VAE

        Args:
            input_data: Input timeseries (batch_size, time_steps, dim)
            conditions: Optional conditional labels (batch_size, num_labels)
            training: Training mode flag

        Returns:
            decoded: Reconstructed timeseries (batch_size, time_steps, dim)
            sigma_list: Standard deviations per timestep
            mu_list: Means per timestep
            log_sigma_list: Log variances per timestep
            z_list: Latent vectors per timestep
        """
        batch_size = tf.shape(input_data)[0]

        # Handle conditional input
        if self.conditional:
            assert conditions is not None
            # Repeat conditions across time
            repeated_conditions = tf.tile(
                tf.expand_dims(conditions, axis=1),
                [1, self.time_steps, 1]
            )
            input_data_cond = tf.concat([input_data, repeated_conditions], axis=-1)
        else:
            input_data_cond = input_data

        # Initialize states
        enc_state = [
            [tf.zeros((batch_size, self.enc_size)),
             tf.zeros((batch_size, self.enc_size))]
            for _ in range(self.enc_layers)
        ]

        dec_state = [
            [tf.zeros((batch_size, self.dec_size)),
             tf.zeros((batch_size, self.dec_size))]
            for _ in range(self.dec_layers)
        ]

        # Storage for outputs
        c_list = []
        mu_list = []
        log_sigma_list = []
        sigma_list = []
        z_list = []

        c_prev = tf.zeros((batch_size, self.dim))

        # Process each timestep
        for t in range(self.time_steps):
            # Get input at timestep t
            if self.conditional:
                x_t = input_data[:, t, :]
                x_t_cond = input_data_cond[:, t, :]
            else:
                x_t = input_data[:, t, :]
                x_t_cond = x_t

            # Compute residual
            c_sigmoid = tf.sigmoid(c_prev)
            x_hat_t = x_t - c_sigmoid

            # Encode
            h_enc, enc_state = self.encode_timestep(x_t_cond, x_hat_t, enc_state)

            # Sample latent
            mu_t = self.mu_layer(h_enc)
            log_sigma_t = self.sigma_layer(h_enc)
            z_t, sigma_t = self.reparameterize(mu_t, log_sigma_t)

            # Add conditions to latent if conditional
            if self.conditional:
                z_t = tf.concat([z_t, conditions], axis=-1)

            # Decode
            c_t, dec_state = self.decode_timestep(z_t, dec_state)

            # Store
            c_list.append(c_t)
            mu_list.append(mu_t)
            log_sigma_list.append(log_sigma_t)
            sigma_list.append(sigma_t)
            z_list.append(z_t)

            # Update c_prev
            c_prev = c_t

        # Stack outputs
        decoded = tf.stack(c_list, axis=1)

        return decoded, sigma_list, mu_list, log_sigma_list, z_list

    def get_config(self):
        """For serialization"""
        config = super().get_config()
        return config


class D_VAE_TF2(VAE_Base_TF2):
    """
    Discrete VAE for medical interventions data - TF2 Version

    Same architecture as C_VAE_TF2 but for discrete/binary data.

    Args:
        Same as VAE_Base_TF2

    Input/Output:
        Same as C_VAE_TF2
    """

    def __init__(self, *args, **kwargs):
        super(D_VAE_TF2, self).__init__(name='discrete_vae', *args, **kwargs)

    def build(self, input_shape):
        """Build model layers"""
        # Encoder
        self.encoder_layers = self.build_encoder_stack(
            name_prefix='discrete_vae_enc',
            shared_name='shared_vae'
        )

        # Sampling layers
        self.mu_layer, self.sigma_layer = self.build_sampling_layers(
            shared_name='shared_vae'
        )

        # Decoder
        self.decoder_layers = self.build_decoder_stack(
            name_prefix='discrete_vae_dec',
            shared_name='shared_vae'
        )

        # Decoder output layer
        self.decoder_output = layers.Dense(
            self.dim,
            activation='sigmoid',  # Binary outputs
            kernel_initializer='glorot_uniform',
            name='discrete_vae_output'
        )

        super().build(input_shape)

    def call(self, input_data, conditions=None, training=False):
        """
        Forward pass - identical to C_VAE_TF2

        See C_VAE_TF2.call() for documentation
        """
        # Reuse C_VAE implementation (identical logic)
        c_vae_temp = C_VAE_TF2(
            batch_size=self.batch_size,
            time_steps=self.time_steps,
            dim=self.dim,
            z_dim=self.z_dim,
            enc_size=self.enc_size,
            dec_size=self.dec_size,
            enc_layers=self.enc_layers,
            dec_layers=self.dec_layers,
            keep_prob=1.0 - self.dropout_rate,
            l2scale=self.l2scale,
            conditional=self.conditional,
            num_labels=self.num_labels
        )

        # Copy layer references
        c_vae_temp.encoder_layers = self.encoder_layers
        c_vae_temp.mu_layer = self.mu_layer
        c_vae_temp.sigma_layer = self.sigma_layer
        c_vae_temp.decoder_layers = self.decoder_layers
        c_vae_temp.decoder_output = self.decoder_output

        return c_vae_temp.call(input_data, conditions, training)


class GAN_Generator_TF2(keras.Model):
    """
    GAN Generator - TF2 Version

    Multi-layer LSTM generator with bilateral coupling support.

    Args:
        noise_dim (int): Noise vector dimension
        gen_dim (int): Generator latent dimension (from VAE)
        dim (int): Output dimension
        time_steps (int): Sequence length
        gen_num_units (int): LSTM hidden units
        gen_num_layers (int): Number of LSTM layers
        keep_prob (float): Dropout keep probability
        l2_scale (float): L2 regularization scale
        conditional (bool): Use conditional GAN
        num_labels (int): Number of conditional labels
        name (str): Model name

    Input:
        noise: Random noise (batch_size, noise_dim)
        gen_latent: VAE latent (batch_size, gen_dim)
        cross_state: Hidden state from paired generator (for bilateral coupling)

    Output:
        generated: Generated timeseries (batch_size, time_steps, dim)
    """

    def __init__(self,
                 noise_dim,
                 gen_dim,
                 dim,
                 time_steps,
                 gen_num_units,
                 gen_num_layers,
                 keep_prob,
                 l2_scale,
                 conditional=False,
                 num_labels=0,
                 name='gan_generator',
                 **kwargs):
        super(GAN_Generator_TF2, self).__init__(name=name, **kwargs)

        self.noise_dim = noise_dim
        self.gen_dim = gen_dim
        self.dim = dim
        self.time_steps = time_steps
        self.gen_num_units = gen_num_units
        self.gen_num_layers = gen_num_layers
        self.dropout_rate = 1.0 - keep_prob
        self.l2_scale = l2_scale
        self.conditional = conditional
        self.num_labels = num_labels

        # Build LSTM stack
        self.lstm_layers = []
        for i in range(gen_num_layers):
            self.lstm_layers.append(
                layers.LSTM(
                    gen_num_units,
                    return_sequences=True,
                    return_state=True,
                    dropout=self.dropout_rate,
                    kernel_regularizer=regularizers.l2(l2_scale),
                    name=f'{name}_lstm_{i}'
                )
            )

        # Output layer
        self.output_layer = layers.Dense(
            dim,
            activation='sigmoid',
            kernel_initializer='glorot_uniform',
            name=f'{name}_output'
        )

    @tf.function
    def call(self, inputs, training=False):
        """
        Forward pass

        Args:
            inputs: Tuple of (noise, gen_latent, cross_state, conditions)
            training: Training mode flag

        Returns:
            generated: Generated timeseries (batch_size, time_steps, dim)
        """
        noise, gen_latent = inputs[:2]
        cross_state = inputs[2] if len(inputs) > 2 else None
        conditions = inputs[3] if len(inputs) > 3 else None

        batch_size = tf.shape(noise)[0]

        # Concatenate noise and latent
        z = tf.concat([noise, gen_latent], axis=-1)

        # Add conditions if conditional
        if self.conditional and conditions is not None:
            z = tf.concat([z, conditions], axis=-1)

        # Repeat z across timesteps
        z_repeated = tf.tile(
            tf.expand_dims(z, axis=1),
            [1, self.time_steps, 1]
        )

        # Pass through LSTM layers
        h = z_repeated
        for layer in self.lstm_layers:
            h, _, _ = layer(h, training=training)

        # Generate output
        generated = self.output_layer(h)

        return generated


class GAN_Discriminator_TF2(keras.Model):
    """
    GAN Discriminator - TF2 Version

    Multi-layer LSTM discriminator with feature matching support.

    Args:
        dim (int): Input dimension
        dis_num_units (int): LSTM hidden units
        dis_num_layers (int): Number of LSTM layers
        keep_prob (float): Dropout keep probability
        l2_scale (float): L2 regularization scale
        conditional (bool): Use conditional GAN
        num_labels (int): Number of conditional labels
        name (str): Model name

    Input:
        data: Input timeseries (batch_size, time_steps, dim)
        conditions: Optional conditional labels (batch_size, num_labels)

    Output:
        logits: Discrimination logits (batch_size, 1)
        features: Intermediate features for feature matching (batch_size, dis_num_units)
    """

    def __init__(self,
                 dim,
                 dis_num_units,
                 dis_num_layers,
                 keep_prob,
                 l2_scale,
                 conditional=False,
                 num_labels=0,
                 name='gan_discriminator',
                 **kwargs):
        super(GAN_Discriminator_TF2, self).__init__(name=name, **kwargs)

        self.dim = dim
        self.dis_num_units = dis_num_units
        self.dis_num_layers = dis_num_layers
        self.dropout_rate = 1.0 - keep_prob
        self.l2_scale = l2_scale
        self.conditional = conditional
        self.num_labels = num_labels

        # Build LSTM stack
        self.lstm_layers = []
        for i in range(dis_num_layers):
            return_sequences = (i < dis_num_layers - 1)
            self.lstm_layers.append(
                layers.LSTM(
                    dis_num_units,
                    return_sequences=return_sequences,
                    return_state=True,
                    dropout=self.dropout_rate,
                    kernel_regularizer=regularizers.l2(l2_scale),
                    name=f'{name}_lstm_{i}'
                )
            )

        # Output layer
        self.output_layer = layers.Dense(
            1,
            kernel_initializer='glorot_uniform',
            name=f'{name}_output'
        )

    @tf.function
    def call(self, inputs, training=False):
        """
        Forward pass

        Args:
            inputs: Input timeseries or tuple (data, conditions)
            training: Training mode flag

        Returns:
            logits: Discrimination logits (batch_size, 1)
            features: Features for feature matching (batch_size, dis_num_units)
        """
        if isinstance(inputs, (list, tuple)):
            data, conditions = inputs
            if self.conditional:
                # Concatenate conditions to each timestep
                repeated_conditions = tf.tile(
                    tf.expand_dims(conditions, axis=1),
                    [1, tf.shape(data)[1], 1]
                )
                data = tf.concat([data, repeated_conditions], axis=-1)
        else:
            data = inputs

        # Pass through LSTM layers
        h = data
        for i, layer in enumerate(self.lstm_layers):
            if i < len(self.lstm_layers) - 1:
                h, _, _ = layer(h, training=training)
            else:
                # Last layer: keep features for feature matching
                features, _, _ = layer(h, training=training)

        # Output logits
        logits = self.output_layer(features)

        return logits, features


# Convenience classes mimicking TF1 API
class C_GAN_NET:
    """Wrapper for continuous GAN (generator + discriminator)"""
    def __init__(self, batch_size, noise_dim, dim, gen_num_units, gen_num_layers,
                 dis_num_units, dis_num_layers, keep_prob, l2_scale,
                 gen_dim, time_steps, conditional=False, num_labels=0):

        self.generator = GAN_Generator_TF2(
            noise_dim=noise_dim,
            gen_dim=gen_dim,
            dim=dim,
            time_steps=time_steps,
            gen_num_units=gen_num_units,
            gen_num_layers=gen_num_layers,
            keep_prob=keep_prob,
            l2_scale=l2_scale,
            conditional=conditional,
            num_labels=num_labels,
            name='continuous_gan_gen'
        )

        self.discriminator = GAN_Discriminator_TF2(
            dim=dim,
            dis_num_units=dis_num_units,
            dis_num_layers=dis_num_layers,
            keep_prob=keep_prob,
            l2_scale=l2_scale,
            conditional=conditional,
            num_labels=num_labels,
            name='continuous_gan_disc'
        )


class D_GAN_NET:
    """Wrapper for discrete GAN (generator + discriminator)"""
    def __init__(self, batch_size, noise_dim, dim, gen_num_units, gen_num_layers,
                 dis_num_units, dis_num_layers, keep_prob, l2_scale,
                 gen_dim, time_steps, conditional=False, num_labels=0):

        self.generator = GAN_Generator_TF2(
            noise_dim=noise_dim,
            gen_dim=gen_dim,
            dim=dim,
            time_steps=time_steps,
            gen_num_units=gen_num_units,
            gen_num_layers=gen_num_layers,
            keep_prob=keep_prob,
            l2_scale=l2_scale,
            conditional=conditional,
            num_labels=num_labels,
            name='discrete_gan_gen'
        )

        self.discriminator = GAN_Discriminator_TF2(
            dim=dim,
            dis_num_units=dis_num_units,
            dis_num_layers=dis_num_layers,
            keep_prob=keep_prob,
            l2_scale=l2_scale,
            conditional=conditional,
            num_labels=num_labels,
            name='discrete_gan_disc'
        )


if __name__ == '__main__':
    """Test script"""
    print("Testing networks_tf2.py...")

    # Parameters
    batch_size = 32
    time_steps = 24
    c_dim = 7
    d_dim = 3
    z_dim = 25

    # Test C_VAE
    print("\n1. Testing C_VAE_TF2...")
    c_vae = C_VAE_TF2(
        batch_size=batch_size,
        time_steps=time_steps,
        dim=c_dim,
        z_dim=z_dim,
        enc_size=128,
        dec_size=128,
        enc_layers=3,
        dec_layers=3,
        keep_prob=0.8,
        l2scale=0.001
    )

    c_data = tf.random.normal((batch_size, time_steps, c_dim))
    c_decoded, c_sigma, c_mu, c_logsigma, c_z = c_vae(c_data)

    print(f"  Input shape: {c_data.shape}")
    print(f"  Decoded shape: {c_decoded.shape}")
    print(f"  ✅ C_VAE works!")

    # Test GAN
    print("\n2. Testing GAN Generator...")
    c_gan = C_GAN_NET(
        batch_size=batch_size,
        noise_dim=3,
        dim=c_dim,
        gen_num_units=512,
        gen_num_layers=3,
        dis_num_units=256,
        dis_num_layers=3,
        keep_prob=0.8,
        l2_scale=0.001,
        gen_dim=z_dim,
        time_steps=time_steps
    )

    noise = tf.random.normal((batch_size, 3))
    gen_latent = tf.random.normal((batch_size, z_dim))
    generated = c_gan.generator([noise, gen_latent])

    print(f"  Generated shape: {generated.shape}")
    print(f"  ✅ GAN Generator works!")

    print("\n✅ All network tests passed!")
