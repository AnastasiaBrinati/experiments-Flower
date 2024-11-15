import tensorflow as tf

# EXPERIMENT 1:
# Class for the LSTM model
class Lstm:
    def __init__(self, learning_rate, sequence_length, num_features, dropout_rate=0.2):
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length  # Number of timesteps in each sequence
        self.num_features = num_features  # Number of features per timestep
        self.dropout_rate = dropout_rate

        # Define the LSTM-based model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.sequence_length, self.num_features)),  # Input shape for LSTM
            tf.keras.layers.LSTM(
                15,
                activation='relu',
                return_sequences=True
            ),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2, activation='linear'))  # Final output layer for each timestep
        ])

        # Define the loss function and optimizer
        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def compile(self):
        # Compile the model with optimizer, loss, and additional metrics
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function,
                           metrics=["cosine_similarity", tf.keras.metrics.MeanAbsoluteError()])

    def get_model(self):
        return self.model