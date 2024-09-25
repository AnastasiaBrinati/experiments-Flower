import tensorflow as tf

# Class for the linear regression model
class Model:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.loss_function = tf.keras.losses.MeanSquaredError()  # Loss for regression
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(12,)),  # Input layer with one feature
            tf.keras.layers.Dense(1)  # Output layer with one unit for regression
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def compile(self):
        # Compile the model with the optimizer and the loss function for regression
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=["cosine_similarity", "mean_absolute_percentage_error"])

    def get_model(self):
        return self.model
