"""
Bilateral LSTM Cell - TensorFlow 2.x Version

Migrated from TF1.x to TF2 using Keras Layer API.
Original: Bilateral_lstm_class.py (TF1.15)
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal


class BilateralLSTMCell(layers.Layer):
    """
    Bilateral LSTM Cell - TF2 Keras Layer

    This custom LSTM cell incorporates cross-dependencies between two modalities
    (e.g., continuous and discrete data) through additional weight matrices (Vi, Vf, Vo, Vc).

    Args:
        hidden_dim (int): Hidden state dimension
        name (str): Layer name (replaces scope_name)
        **kwargs: Additional Keras Layer arguments

    Input:
        x: Input tensor (batch_size, input_dim)
        hidden_memory_tm1: Hidden state from current modality at t-1, tuple of (h, c)
        hidden_memory_tm2: Hidden state from paired modality at t-1, tuple of (h, c)

    Output:
        current_hidden_state: New hidden state (batch_size, hidden_dim)
        state: Tuple of (current_hidden_state, c) for next timestep
    """

    def __init__(self, hidden_dim, name='bilateral_lstm', **kwargs):
        super(BilateralLSTMCell, self).__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim
        self.state_size = (hidden_dim, hidden_dim)  # (h, c)

    def build(self, input_shape):
        """
        Build weights - called automatically when first used

        Args:
            input_shape: Shape of input tensor (batch_size, input_dim)
        """
        if isinstance(input_shape, list):
            # input_shape = [input, state_tm1, state_tm2]
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        self.input_dim = input_dim

        # Initialize with same distribution as TF1 version
        initializer = RandomNormal(mean=0.0, stddev=0.1)

        # Input Gate weights (Wi, Ui, Vi)
        self.Wi = self.add_weight(
            name='Wi',
            shape=(self.input_dim, self.hidden_dim),
            initializer=initializer,
            trainable=True
        )
        self.Ui = self.add_weight(
            name='Ui',
            shape=(self.hidden_dim, self.hidden_dim),
            initializer=initializer,
            trainable=True
        )
        self.Vi = self.add_weight(
            name='Vi',
            shape=(self.hidden_dim, self.hidden_dim),
            initializer=initializer,
            trainable=True
        )

        # Forget Gate weights (Wf, Uf, Vf)
        self.Wf = self.add_weight(
            name='Wf',
            shape=(self.input_dim, self.hidden_dim),
            initializer=initializer,
            trainable=True
        )
        self.Uf = self.add_weight(
            name='Uf',
            shape=(self.hidden_dim, self.hidden_dim),
            initializer=initializer,
            trainable=True
        )
        self.Vf = self.add_weight(
            name='Vf',
            shape=(self.hidden_dim, self.hidden_dim),
            initializer=initializer,
            trainable=True
        )

        # Output Gate weights (Wo, Uo, Vo)
        self.Wo = self.add_weight(
            name='Wo',
            shape=(self.input_dim, self.hidden_dim),
            initializer=initializer,
            trainable=True
        )
        self.Uo = self.add_weight(
            name='Uo',
            shape=(self.hidden_dim, self.hidden_dim),
            initializer=initializer,
            trainable=True
        )
        self.Vo = self.add_weight(
            name='Vo',
            shape=(self.hidden_dim, self.hidden_dim),
            initializer=initializer,
            trainable=True
        )

        # Cell Gate weights (Wc, Uc, Vc)
        self.Wc = self.add_weight(
            name='Wc',
            shape=(self.input_dim, self.hidden_dim),
            initializer=initializer,
            trainable=True
        )
        self.Uc = self.add_weight(
            name='Uc',
            shape=(self.hidden_dim, self.hidden_dim),
            initializer=initializer,
            trainable=True
        )
        self.Vc = self.add_weight(
            name='Vc',
            shape=(self.hidden_dim, self.hidden_dim),
            initializer=initializer,
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs, states):
        """
        Forward pass through the Bilateral LSTM cell

        Args:
            inputs: Tuple of (x, hidden_memory_tm2)
                - x: Input tensor (batch_size, input_dim)
                - hidden_memory_tm2: Cross-modal hidden state from paired modality
            states: List of tensors [hidden_memory_tm1]
                - hidden_memory_tm1: Current modality hidden state at t-1

        Returns:
            current_hidden_state: New hidden state (batch_size, hidden_dim)
            [new_state]: List containing stacked (h, c) for next timestep
        """
        # Unpack inputs
        if isinstance(inputs, (list, tuple)):
            x, hidden_memory_tm2 = inputs
        else:
            x = inputs
            # If no cross-modal state provided, use zeros
            hidden_memory_tm2 = tf.zeros((tf.shape(x)[0], self.hidden_dim * 2))

        # Unpack states
        hidden_memory_tm1 = states[0]

        # Split hidden state and cell state for current modality (tm1)
        previous_hidden_state = hidden_memory_tm1[:, :self.hidden_dim]
        c_prev = hidden_memory_tm1[:, self.hidden_dim:]

        # Split hidden state for paired modality (tm2)
        previous_hidden_state_ = hidden_memory_tm2[:, :self.hidden_dim]

        # Input Gate
        i = tf.sigmoid(
            tf.matmul(x, self.Wi) +
            tf.matmul(previous_hidden_state, self.Ui) +
            tf.matmul(previous_hidden_state_, self.Vi)
        )

        # Forget Gate
        f = tf.sigmoid(
            tf.matmul(x, self.Wf) +
            tf.matmul(previous_hidden_state, self.Uf) +
            tf.matmul(previous_hidden_state_, self.Vf)
        )

        # Output Gate
        o = tf.sigmoid(
            tf.matmul(x, self.Wo) +
            tf.matmul(previous_hidden_state, self.Uo) +
            tf.matmul(previous_hidden_state_, self.Vo)
        )

        # Candidate Cell State
        c_ = tf.tanh(
            tf.matmul(x, self.Wc) +
            tf.matmul(previous_hidden_state, self.Uc) +
            tf.matmul(previous_hidden_state_, self.Vc)
        )

        # New Cell State
        c = f * c_prev + i * c_

        # New Hidden State
        current_hidden_state = o * tf.tanh(c)

        # Stack h and c for next timestep
        new_state = tf.concat([current_hidden_state, c], axis=-1)

        return current_hidden_state, [new_state]

    def get_config(self):
        """For serialization"""
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """For deserialization"""
        return cls(**config)


class MultilayerBilateralLSTM(layers.Layer):
    """
    Stack of Bilateral LSTM Cells - TF2 Keras Layer

    Equivalent to MultilayerCells in TF1 version.

    Args:
        num_layers (int): Number of LSTM layers
        hidden_dim (int): Hidden state dimension for each layer
        name (str): Layer name
        **kwargs: Additional Keras Layer arguments

    Input:
        input: Input tensor (batch_size, input_dim)
        state: List of hidden states for each layer (current modality)
        state_: List of hidden states for each layer (paired modality)

    Output:
        cur_inp: Output from last layer (batch_size, hidden_dim)
        new_states: List of new hidden states for each layer
    """

    def __init__(self, num_layers, hidden_dim, name='multilayer_bilateral_lstm', **kwargs):
        super(MultilayerBilateralLSTM, self).__init__(name=name, **kwargs)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Create cells for each layer
        self.cells = [
            BilateralLSTMCell(hidden_dim, name=f'{name}_cell_{i}')
            for i in range(num_layers)
        ]

    def call(self, inputs, states, states_cross):
        """
        Forward pass through stacked Bilateral LSTM cells

        Args:
            inputs: Input tensor (batch_size, input_dim)
            states: List of states for current modality, one per layer
            states_cross: List of states for paired modality, one per layer

        Returns:
            cur_inp: Output from last layer
            new_states: List of new states for each layer
        """
        cur_inp = inputs
        new_states = []

        for i, cell in enumerate(self.cells):
            # Pass input + cross-modal state to cell
            cur_inp, new_state = cell(
                inputs=(cur_inp, states_cross[i]),
                states=[states[i]]
            )
            new_states.append(new_state[0])

        return cur_inp, new_states

    def get_initial_state(self, batch_size):
        """
        Get zero initial states for all layers

        Args:
            batch_size (int): Batch size

        Returns:
            List of initial states (zeros) for each layer
        """
        return [
            tf.zeros((batch_size, self.hidden_dim * 2))
            for _ in range(self.num_layers)
        ]

    def get_config(self):
        """For serialization"""
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'hidden_dim': self.hidden_dim,
        })
        return config


# Convenience function for creating bilateral LSTM
def create_bilateral_lstm(num_layers, hidden_dim, name='bilateral_lstm'):
    """
    Factory function to create a MultilayerBilateralLSTM

    Args:
        num_layers (int): Number of stacked layers
        hidden_dim (int): Hidden dimension for each layer
        name (str): Name prefix for the layers

    Returns:
        MultilayerBilateralLSTM instance

    Example:
        >>> lstm = create_bilateral_lstm(num_layers=3, hidden_dim=128)
        >>> batch_size = 32
        >>> states_tm1 = lstm.get_initial_state(batch_size)
        >>> states_tm2 = lstm.get_initial_state(batch_size)
        >>> x = tf.random.normal((batch_size, 10))
        >>> output, new_states = lstm(x, states_tm1, states_tm2)
    """
    return MultilayerBilateralLSTM(num_layers, hidden_dim, name=name)


if __name__ == '__main__':
    """
    Test script to verify Bilateral LSTM functionality
    """
    print("Testing Bilateral LSTM Cell (TF2)...")

    # Parameters
    batch_size = 32
    input_dim = 10
    hidden_dim = 128
    num_layers = 3

    # Create multilayer bilateral LSTM
    lstm = create_bilateral_lstm(num_layers=num_layers, hidden_dim=hidden_dim)

    # Initialize states
    states_tm1 = lstm.get_initial_state(batch_size)
    states_tm2 = lstm.get_initial_state(batch_size)

    # Random input
    x = tf.random.normal((batch_size, input_dim))

    # Forward pass
    output, new_states = lstm(x, states_tm1, states_tm2)

    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Number of states: {len(new_states)}")
    print(f"✅ State shape: {new_states[0].shape}")

    # Verify shapes
    assert output.shape == (batch_size, hidden_dim), f"Output shape mismatch: {output.shape}"
    assert len(new_states) == num_layers, f"Number of states mismatch: {len(new_states)}"
    assert new_states[0].shape == (batch_size, hidden_dim * 2), f"State shape mismatch: {new_states[0].shape}"

    print("\n✅ All tests passed! Bilateral LSTM TF2 migration successful.")
