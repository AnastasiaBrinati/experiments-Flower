import logging
import numpy as np
import tensorflow as tf
from flwr_datasets import FederatedDataset

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module


def load_data(data_sampling_percentage=0.5, client_id=1, total_clients=2):
    """Load federated dataset partition based on client ID.

    Args:
        data_sampling_percentage (float): Percentage of the dataset to use for training.
        client_id (int): Unique ID for the client.
        total_clients (int): Total number of clients.

    Returns:
        Tuple of arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    # Download and partition dataset
    fds = FederatedDataset(dataset="anastasiafrosted/func", partitioners={"train": total_clients})
    partition = fds.load_partition(client_id - 1, "train")
    partition.set_format("numpy")

    # Divide data on each client: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2, seed=42)

    # Extract the relevant columns
    avg_duration_train = partition["train"]["AverageDuration"]
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

    avg_duration_test = partition["test"]["AverageDuration"]
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
        avg_duration_train, count_for_duration_train, avg_allocated_mb_train, count_for_memory_train, minute_train,
        trigger_others_train, trigger_orchestration_train, trigger_event_train, trigger_timer_train, trigger_queue_train,
        trigger_storage_train, trigger_http_train
    ))

    x_test = np.column_stack((
        avg_duration_test, count_for_duration_test, avg_allocated_mb_test, count_for_memory_test, minute_test,
        trigger_others_test, trigger_orchestration_test, trigger_event_test, trigger_timer_test, trigger_queue_test,
        trigger_storage_test, trigger_http_test
    ))

    # The label is 'invocations'
    y_train = partition["train"]["invocations"]
    y_test = partition["test"]["invocations"]

    # Apply data sampling
    num_samples = int(data_sampling_percentage * len(x_train))
    indices = np.random.choice(len(x_train), num_samples, replace=False)
    x_train, y_train = x_train[indices], y_train[indices]

    return (x_train, y_train), (x_test, y_test)
