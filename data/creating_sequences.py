from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import row_number, collect_list, floor, size, col, to_timestamp

# Initialize Spark session if needed
spark = SparkSession.builder.appName("SequenceCreation").getOrCreate()

# Load data and assume the CSV file includes an 'endpoint_uuid', 'date', 'feature1', 'feature2', and 'target' columns
train_data = spark.read.csv("scaled_data/globus/globus_split/train/globus_train.csv", header=True, inferSchema=True)
test_data = spark.read.csv("scaled_data/globus/globus_split/test/globus_test.csv", header=True, inferSchema=True)

# Ensure 'date' is a timestamp
# data = data.withColumn("date", to_timestamp("date"))

# Define the sequence length
sequence_length = 7

# Step 1: Partition by 'id' and order by 'date'
window = Window.partitionBy("endpoint_uuid").orderBy("timestamp")

# Step 2: Create a row number within each partition, ensuring sequential order within each 'id'
train_data = train_data.withColumn("row_num", row_number().over(window))
test_data = test_data.withColumn("row_num", row_number().over(window))

# Step 3: Create a sequence ID by dividing row_num by the sequence length, grouping rows into blocks of 7 steps
train_data = train_data.withColumn("sequence_id", floor((col("row_num") - 1) / sequence_length))
test_data = test_data.withColumn("sequence_id", floor((col("row_num") - 1) / sequence_length))

# Step 4: Aggregate to create sequences of 7 steps by collecting lists of each feature
df_train_sequences = (
    train_data.groupBy("endpoint_uuid", "sequence_id")
    .agg(
        collect_list("timestamp").alias("timestamp_seq"),
        collect_list("date").alias("date_seq"),  # Include date in the sequence
        collect_list("hour").alias("hour_seq"),
        collect_list("minute").alias("minute_seq"),
        collect_list("second").alias("second_seq"),

        collect_list("assignment_time").alias("assignment_time_seq"),
        collect_list("scheduling_time").alias("scheduling_time_seq"),
        collect_list("queue_time").alias("queue_time_seq"),
        collect_list("results_time").alias("results_time_seq"),
        #collect_list("execution_time").alias("execution_time_seq"),
        collect_list("total_execution_time").alias("total_execution_time_seq"),

        collect_list("argument_size").alias("argument_size_seq"),
        collect_list("loc").alias("loc_seq"),
        #collect_list("cyc_complexity").alias("cyc_complexity_seq"),
        collect_list("num_of_imports").alias("num_of_imports_seq"),

        collect_list("e_type_LSFProvider").alias("e_type_LSFProvider_seq"),
        collect_list("e_type_CobaltProvider").alias("e_type_CobaltProvider_seq"),
        collect_list("e_type_PBSProProvider").alias("e_type_PBSProProvider_seq"),
        collect_list("e_type_LocalProvider").alias("e_type_LocalProvider_seq"),
        collect_list("e_type_KubernetesProvider").alias("e_type_KubernetesProvider_seq"),
        collect_list("e_type_SlurmProvider").alias("e_type_SlurmProvider_seq"),

        # EXPERIMENT 1: CPU-intensive apps -> focus on execution time and cyclomatic complexity
        collect_list("execution_time").alias("execution_time_seq"),
        collect_list("cyc_complexity").alias("cyc_complexity_seq")
    )
)

df_test_sequences = (
    test_data.groupBy("endpoint_uuid", "sequence_id")
    .agg(
        collect_list("timestamp").alias("timestamp_seq"),
        collect_list("date").alias("date_seq"),  # Include date in the sequence
        collect_list("hour").alias("hour_seq"),
        collect_list("minute").alias("minute_seq"),
        collect_list("second").alias("second_seq"),

        collect_list("assignment_time").alias("assignment_time_seq"),
        collect_list("scheduling_time").alias("scheduling_time_seq"),
        collect_list("queue_time").alias("queue_time_seq"),
        collect_list("results_time").alias("results_time_seq"),
        #collect_list("execution_time").alias("execution_time_seq"),
        collect_list("total_execution_time").alias("total_execution_time_seq"),

        collect_list("argument_size").alias("argument_size_seq"),
        collect_list("loc").alias("loc_seq"),
        #collect_list("cyc_complexity").alias("cyc_complexity_seq"),
        collect_list("num_of_imports").alias("num_of_imports_seq"),

        collect_list("e_type_LSFProvider").alias("e_type_LSFProvider_seq"),
        collect_list("e_type_CobaltProvider").alias("e_type_CobaltProvider_seq"),
        collect_list("e_type_PBSProProvider").alias("e_type_PBSProProvider_seq"),
        collect_list("e_type_LocalProvider").alias("e_type_LocalProvider_seq"),
        collect_list("e_type_KubernetesProvider").alias("e_type_KubernetesProvider_seq"),
        collect_list("e_type_SlurmProvider").alias("e_type_SlurmProvider_seq"),

        # EXPERIMENT 1: CPU-intensive apps -> focus on execution time and cyclomatic complexity
        collect_list("execution_time").alias("execution_time_seq"),
        collect_list("cyc_complexity").alias("cyc_complexity_seq")
    )
)

# Step 5: Filter out incomplete sequences with less than 7 steps
df_train_sequences = df_train_sequences.filter(size(col("execution_time_seq")) == sequence_length)
df_train_sequences = df_train_sequences.filter(size(col("cyc_complexity_seq")) == sequence_length)

df_test_sequences = df_test_sequences.filter(size(col("execution_time_seq")) == sequence_length)
df_test_sequences = df_test_sequences.filter(size(col("cyc_complexity_seq")) == sequence_length)

# Show the resulting sequences for verification
# df_train_sequences.show(truncate=False)

# Collect to Pandas DataFrame
df_train_sequences = df_train_sequences.orderBy(["sequence_id", "timestamp_seq"])
df_train_sequences_pd = df_train_sequences.toPandas()
df_test_sequences = df_test_sequences.orderBy(["sequence_id", "timestamp_seq"])
df_test_sequences_pd = df_test_sequences.toPandas()

# Stop the Spark session
spark.stop()

from datasets import Dataset, DatasetDict

# Convert to Hugging Face Dataset
hf_train_dataset = Dataset.from_pandas(df_train_sequences_pd)
hf_test_dataset = Dataset.from_pandas(df_test_sequences_pd)

# Split into train and test sets (e.g., 80-20 split)
#train_test_split = hf_dataset.train_test_split(test_size=0.2)
dataset_dict = DatasetDict({
    'train': hf_train_dataset,
    'test': hf_test_dataset
})

# Save the DatasetDict to a local directory
dataset_dict.save_to_disk("my_sequences_dataset")

dataset_dict.push_to_hub("anastasiafrosted/my_sequences_dataset")
