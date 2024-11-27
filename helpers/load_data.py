import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from flwr_datasets import FederatedDataset

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

def load_data_for_arima(client_id=1, total_clients=2):
    # Dowloading directly endpoint x client
    dataset_path = "anastasiafrosted/endpoint" + str(client_id - 1)

    # Download and partition dataset
    #fds_train = FederatedDataset(dataset=dataset_path, partitioners={"train": total_clients})
    #partition_train = fds_train.load_partition(client_id - 1, "train")
    #partition_train.set_format("numpy")

    fds_test = FederatedDataset(dataset=dataset_path, partitioners={"test": total_clients})
    partition_test = fds_test.load_partition(client_id - 1, "test")
    partition_test.set_format("numpy")
    df = pd.DataFrame(partition_test)
    df.drop("endpoint_uuid", inplace=True, axis=1)
    df['timestamp'] = df['timestamp'].str[:-9]
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S.')
    df = df.dropna()
    df = df.sort_values("timestamp").set_index('timestamp')
    df = df.dropna()
    df = df.asfreq('s').interpolate(method='time')

    origin, start_date, end_date = "2022-12-20 00:00:00", "2023-01-06 00:00:00", "2023-07-03 23:00:00"
    time_period = pd.date_range(start_date, end_date, freq='s')

    input_features = [
        "invocations_per_hour", "avg_argument_size", "avg_loc", "avg_num_of_imports", "avg_cyc_complexity",
        "e_type_LSFProvider", "e_type_CobaltProvider", "e_type_PBSProProvider", "e_type_LocalProvider",
        "e_type_KubernetesProvider", "e_type_SlurmProvider"
    ]

    exogenous_features = df[input_features].iloc[-4000:]
    target_1 = partition_test["avg_total_execution_time"][-4000:]
    target_2 = partition_test["avg_system_processing_time"][-4000:]

    return (exogenous_features, target_1, target_2)


def load_data_for_lstm(client_id=1, total_clients=2):
    """Load federated dataset partition based on client ID.

    Args:
        data_sampling_percentage (float): Percentage of the dataset to use for training.
        client_id (int): Unique ID for the client.
        total_clients (int): Total number of clients.

    Returns:
        Tuple of arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    # Dowloading directly endpoint x client
    dataset_path = "anastasiafrosted/sequenced_endpoint_"+str(client_id-1)

    # Download and partition dataset
    fds_train = FederatedDataset(dataset=dataset_path, partitioners={"train": total_clients})
    partition_train = fds_train.load_partition(client_id - 1, "train")
    partition_train.set_format("numpy")

    fds_test = FederatedDataset(dataset=dataset_path, partitioners={"test": total_clients})
    partition_test = fds_test.load_partition(client_id - 1, "test")
    partition_test.set_format("numpy")

    # Extract features for training and testing sets
    input_features = [
        "invocations_per_hour_seq",
        "avg_argument_size_seq", "avg_loc_seq", "avg_num_of_imports_seq", "avg_cyc_complexity_seq",
    ]

    timestamps_columns = [
        "timestamp_seq", "timestamp_target"
    ]

    target_features = ["avg_total_execution_time_target", "avg_system_processing_time_target"]

    # THIS IS WHERE IT CHANGES FOR THE MODEL!!!!

    # Convert feature columns to a 3D numpy array (num_samples, sequence_length, num_features)
    x_train = np.stack([partition_train[col] for col in input_features], axis=-1)
    x_test = np.stack([partition_test[col] for col in input_features], axis=-1)
    # this is being added for the grap
    x_test_timestamps = partition_test[timestamps_columns[1]]

    # Stack target features
    y_train = np.stack([partition_train[tar] for tar in target_features], axis=-1)
    y_test = np.stack([partition_test[tar] for tar in target_features], axis=-1)

    return (x_train, y_train), (x_test, x_test_timestamps, y_test)