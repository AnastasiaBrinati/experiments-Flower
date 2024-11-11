import logging
import numpy as np
import tensorflow as tf
from flwr_datasets import FederatedDataset

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module


def load_azure_data(data_sampling_percentage=0.5, client_id=1, total_clients=2):
    """Load federated dataset partition based on client ID.

    Args:
        data_sampling_percentage (float): Percentage of the dataset to use for training.
        client_id (int): Unique ID for the client.
        total_clients (int): Total number of clients.

    Returns:
        Tuple of arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    # Download and partition dataset
    fds = FederatedDataset(dataset="anastasiafrosted/azure_prova", partitioners={"train": total_clients})
    partition = fds.load_partition(client_id - 1, "train")
    partition.set_format("numpy")

    # Divide data on each client: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2, seed=42)

    # Extract the relevant columns
    invocations_train = partition["train"]["invocations"]
    count_for_duration_train = partition["train"]["CountForDuration"]
    avg_allocated_mb_train = partition["train"]["AverageAllocatedMb"]
    count_for_memory_train = partition["train"]["CountForMemory"]
    minute_train = partition["train"]["minute"]
    trigger_others_train = partition["train"]["Trigger_others"]
    trigger_orchestration_train = partition["train"]["Trigger_orchestration"]
    trigger_event_train = partition["train"]["Trigger_event"]
    trigger_timer_train = partition["train"]["Trigger_timer"]
    trigger_queue_train = partition["train"]["Trigger_queue"]
    trigger_storage_train = partition["train"]["Trigger_storage"]
    trigger_http_train = partition["train"]["Trigger_http"]

    invocations_test = partition["test"]["invocations"]
    count_for_duration_test = partition["test"]["CountForDuration"]
    avg_allocated_mb_test = partition["test"]["AverageAllocatedMb"]
    count_for_memory_test = partition["test"]["CountForMemory"]
    minute_test = partition["test"]["minute"]
    trigger_others_test = partition["test"]["Trigger_others"]
    trigger_orchestration_test = partition["test"]["Trigger_orchestration"]
    trigger_event_test = partition["test"]["Trigger_event"]
    trigger_timer_test = partition["test"]["Trigger_timer"]
    trigger_queue_test = partition["test"]["Trigger_queue"]
    trigger_storage_test = partition["test"]["Trigger_storage"]
    trigger_http_test = partition["test"]["Trigger_http"]

    # Combine all features into a single array with multiple features per sample (for both training and test sets)
    x_train = np.column_stack((
        invocations_train, count_for_duration_train, avg_allocated_mb_train, count_for_memory_train, minute_train,
        trigger_others_train, trigger_orchestration_train, trigger_event_train, trigger_timer_train, trigger_queue_train,
        trigger_storage_train, trigger_http_train
    ))

    x_test = np.column_stack((
        invocations_test, count_for_duration_test, avg_allocated_mb_test, count_for_memory_test, minute_test,
        trigger_others_test, trigger_orchestration_test, trigger_event_test, trigger_timer_test, trigger_queue_test,
        trigger_storage_test, trigger_http_test
    ))

    # The label is 'invocations'
    y_train = partition["train"]["AverageDuration"]
    y_test = partition["test"]["AverageDuration"]

    # Apply data sampling
    num_samples = int(data_sampling_percentage * len(x_train))
    indices = np.random.choice(len(x_train), num_samples, replace=False)
    x_train, y_train = x_train[indices], y_train[indices]

    return (x_train, y_train), (x_test, y_test)

def load_globus_data(data_sampling_percentage=0.5, client_id=1, total_clients=2):
    """Load federated dataset partition based on client ID.

    Args:
        data_sampling_percentage (float): Percentage of the dataset to use for training.
        client_id (int): Unique ID for the client.
        total_clients (int): Total number of clients.

    Returns:
        Tuple of arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    # Download and partition dataset
    fds_train = FederatedDataset(dataset="anastasiafrosted/my_sequences_dataset", partitioners={"train": total_clients})
    partition_train = fds_train.load_partition(client_id - 1, "train")
    partition_train.set_format("numpy")

    # Divide data on each client: 80% train, 20% test
    #partition = partition.train_test_split(test_size=0.2, seed=42)
    fds_test = FederatedDataset(dataset="anastasiafrosted/my_sequences_dataset", partitioners={"test": total_clients})
    partition_test = fds_test.load_partition(client_id - 1, "test")
    partition_test.set_format("numpy")

    # Extract features for training and testing sets
    feature_columns = [
        "hour_seq", "minute_seq", "second_seq", "assignment_time_seq",
        "scheduling_time_seq", "queue_time_seq", "results_time_seq", "total_execution_time_seq",
        "argument_size_seq", "loc_seq", "num_of_imports_seq",
        "e_type_LSFProvider_seq", "e_type_CobaltProvider_seq", "e_type_PBSProProvider_seq",
        "e_type_LocalProvider_seq", "e_type_KubernetesProvider_seq", "e_type_SlurmProvider_seq",
        # TARGET 1: "execution_time_seq",
        # TARGET 2: "cyc_complexity_seq"
    ]

    timestamps_columns = [
        "timestamp_seq", "date_seq"
    ]

    # THIS IS WHERE IT CHANGES !!!!
    # Convert feature columns to a 3D numpy array (num_samples, sequence_length, num_features)
    x_train = np.stack([partition_train[col] for col in feature_columns], axis=-1)
    x_test = np.stack([partition_test[col] for col in feature_columns], axis=-1)
    #logger.info("   CREATING x_test_timestamps...")
    x_test_timestamps = np.stack([partition_test[col].astype(str) for col in timestamps_columns], axis=-1)
    #logger.info("   CREATED x_test_timestamps...")

    # The label is 'total_execution_time'
    y_train = np.stack([partition_train[tar] for tar in ["execution_time_seq", "cyc_complexity_seq"]], axis=-1)
    y_test = np.stack([partition_test[tar] for tar in ["execution_time_seq", "cyc_complexity_seq"]], axis=-1)

    # Apply data sampling
    num_samples = int(data_sampling_percentage * len(x_train))
    indices = np.random.choice(len(x_train), num_samples, replace=False)
    x_train, y_train = x_train[indices], y_train[indices]

    #logger.info("   RETURNING (x_train, y_train), (x_test, x_test_timestamps, y_test)...")
    return (x_train, y_train), (x_test, x_test_timestamps, y_test)