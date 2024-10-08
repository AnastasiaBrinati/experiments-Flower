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
    fds = FederatedDataset(dataset="anastasiafrosted/globus_prova", partitioners={"train": total_clients})
    partition = fds.load_partition(client_id - 1, "train")
    partition.set_format("numpy")

    # Divide data on each client: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2, seed=42)

    # Extract the relevant columns
    assignment_time = partition["train"]["assignment_time"]
    scheduling_time = partition["train"]["scheduling_time"]
    queue_time = partition["train"]["queue_time"]
    execution_time = partition["train"]["execution_time"]

    argument_size = partition["train"]["argument_size"]
    loc = partition["train"]["loc"]
    cyc_complexity = partition["train"]["cyc_complexity"]
    num_of_imports = partition["train"]["num_of_imports"]

    e_type_LSFProvider = partition["train"]["e_type_LSFProvider"]
    e_type_CobaltProvider = partition["train"]["e_type_CobaltProvider"]
    e_type_PBSProProvider = partition["train"]["e_type_PBSProProvider"]
    e_type_LocalProvider = partition["train"]["e_type_LocalProvider"]
    e_type_KubernetesProvider = partition["train"]["e_type_KubernetesProvider"]
    e_type_SlurmProvider = partition["train"]["e_type_SlurmProvider"]

    e_vers_2_0_1 = partition["train"]["e_vers_2_0_1"]
    e_vers_1_0_8 = partition["train"]["e_vers_1_0_8"]
    e_vers_1_0_13 = partition["train"]["e_vers_1_0_13"]
    e_vers_2_1_0a1 = partition["train"]["e_vers_2_1_0a1"]
    e_vers_1_0_11a1 = partition["train"]["e_vers_1_0_11a1"]
    e_vers_1_0_12 = partition["train"]["e_vers_1_0_12"]
    e_vers_2_0_2a0 = partition["train"]["e_vers_2_0_2a0"]
    e_vers_2_1_0 = partition["train"]["e_vers_2_1_0"]
    e_vers_1_0_10 = partition["train"]["e_vers_1_0_10"]
    e_vers_2_0_0 = partition["train"]["e_vers_2_0_0"]
    e_vers_2_0_0a0 = partition["train"]["e_vers_2_0_0a0"]
    e_vers_2_0_3 = partition["train"]["e_vers_2_0_3"]
    e_vers_2_0_2 = partition["train"]["e_vers_2_0_2"]
    e_vers_2_0_1a0 = partition["train"]["e_vers_2_0_1a0"]
    e_vers_2_0_1a1 = partition["train"]["e_vers_2_0_1a1"]
    e_vers_1_0_11 = partition["train"]["e_vers_1_0_11"]


#########################

    assignment_time_test = partition["test"]["assignment_time"]
    scheduling_time_test = partition["test"]["scheduling_time"]
    queue_time_test = partition["test"]["queue_time"]
    execution_time_test = partition["test"]["execution_time"]

    argument_size_test = partition["test"]["argument_size"]
    loc_test = partition["test"]["loc"]
    cyc_complexity_test = partition["test"]["cyc_complexity"]
    num_of_imports_test = partition["test"]["num_of_imports"]

    e_type_LSFProvider_test = partition["test"]["e_type_LSFProvider"]
    e_type_CobaltProvider_test = partition["test"]["e_type_CobaltProvider"]
    e_type_PBSProProvider_test = partition["test"]["e_type_PBSProProvider"]
    e_type_LocalProvider_test = partition["test"]["e_type_LocalProvider"]
    e_type_KubernetesProvider_test = partition["test"]["e_type_KubernetesProvider"]
    e_type_SlurmProvider_test = partition["test"]["e_type_SlurmProvider"]

    e_vers_2_0_1_test = partition["test"]["e_vers_2_0_1"]
    e_vers_1_0_8_test = partition["test"]["e_vers_1_0_8"]
    e_vers_1_0_13_test = partition["test"]["e_vers_1_0_13"]
    e_vers_2_1_0a1_test = partition["test"]["e_vers_2_1_0a1"]
    e_vers_1_0_11a1_test = partition["test"]["e_vers_1_0_11a1"]
    e_vers_1_0_12_test = partition["test"]["e_vers_1_0_12"]
    e_vers_2_0_2a0_test = partition["test"]["e_vers_2_0_2a0"]
    e_vers_2_1_0_test = partition["test"]["e_vers_2_1_0"]
    e_vers_1_0_10_test = partition["test"]["e_vers_1_0_10"]
    e_vers_2_0_0_test = partition["test"]["e_vers_2_0_0"]
    e_vers_2_0_0a0_test = partition["test"]["e_vers_2_0_0a0"]
    e_vers_2_0_3_test = partition["test"]["e_vers_2_0_3"]
    e_vers_2_0_2_test = partition["test"]["e_vers_2_0_2"]
    e_vers_2_0_1a0_test = partition["test"]["e_vers_2_0_1a0"]
    e_vers_2_0_1a1_test = partition["test"]["e_vers_2_0_1a1"]
    e_vers_1_0_11_test = partition["test"]["e_vers_1_0_11"]



    # Combine all features into a single array with multiple features per sample (for both training and test sets)
    x_train = np.column_stack((
        assignment_time, scheduling_time,
        queue_time, execution_time,
        argument_size, loc, cyc_complexity, num_of_imports,
        e_type_LSFProvider, e_type_CobaltProvider, e_type_PBSProProvider,
        e_type_LocalProvider, e_type_KubernetesProvider, e_type_SlurmProvider,
        e_vers_2_0_1, e_vers_1_0_8, e_vers_1_0_13, e_vers_2_1_0a1,
        e_vers_1_0_11a1, e_vers_1_0_12, e_vers_2_0_2a0, e_vers_2_1_0,
        e_vers_1_0_10, e_vers_2_0_0, e_vers_2_0_0a0, e_vers_2_0_3,
        e_vers_2_0_2, e_vers_2_0_1a0, e_vers_2_0_1a1, e_vers_1_0_11
    ))

    x_test = np.column_stack((
        assignment_time_test, scheduling_time_test,
        queue_time_test, execution_time_test,
        argument_size_test, loc_test, cyc_complexity_test, num_of_imports_test,
        e_type_LSFProvider_test, e_type_CobaltProvider_test, e_type_PBSProProvider_test,
        e_type_LocalProvider_test, e_type_KubernetesProvider_test, e_type_SlurmProvider_test,
        e_vers_2_0_1_test, e_vers_1_0_8_test, e_vers_1_0_13_test, e_vers_2_1_0a1_test,
        e_vers_1_0_11a1_test, e_vers_1_0_12_test, e_vers_2_0_2a0_test, e_vers_2_1_0_test,
        e_vers_1_0_10_test, e_vers_2_0_0_test, e_vers_2_0_0a0_test, e_vers_2_0_3_test,
        e_vers_2_0_2_test, e_vers_2_0_1a0_test, e_vers_2_0_1a1_test, e_vers_1_0_11_test

    ))

    # The label is 'invocations'
    y_train = partition["train"]["total_execution_time"]
    y_test = partition["test"]["total_execution_time"]

    # Apply data sampling
    num_samples = int(data_sampling_percentage * len(x_train))
    indices = np.random.choice(len(x_train), num_samples, replace=False)
    x_train, y_train = x_train[indices], y_train[indices]

    return (x_train, y_train), (x_test, y_test)