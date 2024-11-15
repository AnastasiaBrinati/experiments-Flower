import argparse
import logging
import os
import pandas as pd
import numpy as np

import flwr as fl
import tensorflow as tf
from helpers.load_data import load_azure_data
from helpers.load_data import load_globus_data

from model.model import Model
from model.lstm import Lstm

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the modulef

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower client")

parser.add_argument(
    "--server_address", type=str, default="server:8080", help="Address of the server"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training"
)
parser.add_argument(
    "--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer"
)
parser.add_argument("--client_id", type=int, default=1, help="Unique ID for the client")
parser.add_argument(
    "--total_clients", type=int, default=2, help="Total number of clients"
)
parser.add_argument(
    "--data_percentage", type=float, default=0.5, help="Portion of client data to use"
)

args = parser.parse_args()

# Create an instance of the model and pass the learning rate as an argument
# BASE
#model = Model(learning_rate=args.learning_rate, num_features=30)
# LSTM
model = Lstm(learning_rate=args.learning_rate, sequence_length=7,num_features=11)

# Compile the model
model.compile()


class Client(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args

        logger.info("Preparing data...")
        (x_train, y_train), (x_test, x_test_timestamps, y_test) = load_globus_data(
            data_sampling_percentage=self.args.data_percentage,
            client_id=self.args.client_id,
            total_clients=self.args.total_clients,
        )

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.x_test_timestamps = x_test_timestamps
        self.y_test = y_test

    def get_parameters(self, config):
        return model.get_model().get_weights()

    def fit(self, parameters, config):
        model.get_model().set_weights(parameters)

        history = model.get_model().fit(
            self.x_train, self.y_train, batch_size=self.args.batch_size
        )

        results = {
            "cosine_similarity": float(history.history["cosine_similarity"][-1]),
            "mean_absolute_error": float(history.history["mean_absolute_error"][-1]),
        }

        parameters_prime = model.get_model().get_weights()
        return parameters_prime, len(self.x_train), results

    def evaluate(self, parameters, config):
        model.get_model().set_weights(parameters)

        # Get predictions
        predictions = model.get_model().predict(self.x_test)

        # Step 1: Take the predicted and actual target features
        predicted_execution_time = predictions[:, :, 0]  # First target
        actual_execution_time = self.x_test[:, :, 0]
        predicted_cyc_complexity = predictions[:, :, 1]  # Second target
        actual_cyc_complexity = self.x_test[:, :, 1]

        # now apply a -most-recent-prediction- strategy
        # Step 2: Take the first value of each sequence and stop before the last one
        predicted_execution_time = np.array([seq[0] for seq in predicted_execution_time])
        predicted_cyc_complexity = np.array([seq[0] for seq in predicted_cyc_complexity])

        # we do the same on the actual values just to reuse code but the values of course are repeated on rows
        actual_execution_time = np.array([seq[0] for seq in actual_execution_time])
        actual_cyc_complexity = np.array([seq[0] for seq in actual_cyc_complexity])

        # Organize the timestamps with the same order as the predictions:
        timestamps = np.array([timestamps[0] for timestamps in self.x_test_timestamps])

        # Combine predictions and actual values
        data = {
            "timestamps": timestamps,
            "actual_execution_time": actual_execution_time,
            "predicted_execution_time": predicted_execution_time,
            "actual_cyc_complexity":  actual_cyc_complexity,
            "predicted_cyc_complexity": predicted_cyc_complexity,
        }

        # Convert to a Pandas DataFrame
        df = pd.DataFrame(data)

        # Save to a CSV file
        output_file = f"client_{self.args.client_id}_predictions.csv"
        df.to_csv(output_file, index=False)
        logger.info("Predictions saved to %s", output_file)

        # Calculate evaluation metrics
        metrics = model.get_model().evaluate(
            self.x_test, self.y_test, batch_size=self.args.batch_size
        )

        # Return the loss, number of examples evaluated on, and metrics
        return float(metrics[0]), len(self.x_test), {"cos": float(metrics[1]), "mae": float(metrics[2])}


# Function to Start the Client
def start_fl_client():
    try:
        client = Client(args).to_client()
        fl.client.start_client(server_address=args.server_address, client=client)
        logger.info("   Ended rounds...")
    except Exception as e:
        logger.error("Error starting FL client: %s", e)
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Call the function to start the client
    start_fl_client()
