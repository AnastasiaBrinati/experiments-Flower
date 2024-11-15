from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import row_number, collect_list, col, size, slice
from collections import defaultdict
from datasets import Dataset, DatasetDict
from pyspark.sql.types import IntegerType

# Initialize Spark session
spark = SparkSession.builder.appName("LSTMSequencePreparation").getOrCreate()

# Load data and assume the CSV file includes the required columns
train_data = spark.read.csv("scaled_data/globus/endpoints/endpoint2/split/train/endpoint2_train.csv", header=True, inferSchema=True)
test_data = spark.read.csv("scaled_data/globus/endpoints/endpoint2/split/test/endpoint2_test.csv", header=True, inferSchema=True)

# Step 0: Sequence and prediction lengths
sequence_length = 7
prediction_length = 7
total_length = sequence_length + prediction_length

# Step 1: Partition by 'timestamp' and order by 'timestamp'
window = Window.orderBy("timestamp")

# Step 2: Create a row number for sequential ordering
train_data = train_data.withColumn("row_num", row_number().over(window))
test_data = test_data.withColumn("row_num", row_number().over(window))

# Define input and output features
input_features = ["timestamp",
                  "invocations_per_hour", "avg_argument_size", "avg_loc", "avg_cyc_complexity", "avg_num_of_imports",
                  "e_type_LSFProvider", "e_type_CobaltProvider", "e_type_PBSProProvider",
                  "e_type_LocalProvider", "e_type_KubernetesProvider", "e_type_SlurmProvider"]
output_features = ["timestamp", "avg_execution_time", "avg_scheduling_time"]

# Step 3: Create sliding window sequences for input and target features
sliding_window = Window.orderBy("row_num").rowsBetween(0, total_length - 1)

# Collect input feature sequences
for feature in input_features + output_features:
    train_data = train_data.withColumn(f"{feature}_seq", collect_list(col(feature)).over(sliding_window))
    test_data = test_data.withColumn(f"{feature}_seq", collect_list(col(feature)).over(sliding_window))

# Step 4: Filter for rows with full sequences
train_data = train_data.filter(size(col(f"{input_features[0]}_seq")) == total_length)
test_data = test_data.filter(size(col(f"{input_features[0]}_seq")) == total_length)

# Step 5: Extract input and target sequences
for feature in input_features:
    train_data = train_data.withColumn(f"{feature}_seq_input", slice(col(f"{feature}_seq"), 1, sequence_length))
    test_data = test_data.withColumn(f"{feature}_seq_input", slice(col(f"{feature}_seq"), 1, sequence_length))

for feature in output_features:
    train_data = train_data.withColumn(f"{feature}_seq_target", slice(col(f"{feature}_seq"), sequence_length + 1, prediction_length))
    test_data = test_data.withColumn(f"{feature}_seq_target",
                                       slice(col(f"{feature}_seq"), sequence_length + 1, prediction_length))

# Step 6: Select only the required columns for input and target
train_data = train_data.select(
    *[col(f"{feature}_seq_input").alias(f"{feature}_seq") for feature in input_features],
    *[col(f"{feature}_seq_target").alias(f"{feature}_target") for feature in output_features]
)
test_data = test_data.select(
    *[col(f"{feature}_seq_input").alias(f"{feature}_seq") for feature in input_features],
    *[col(f"{feature}_seq_target").alias(f"{feature}_target") for feature in output_features]
)

# Step 7: Convert Spark DataFrame to a list of dictionaries
train_data_list = train_data.rdd.map(lambda row: row.asDict()).collect()
test_data_list = test_data.rdd.map(lambda row: row.asDict()).collect()

# Step 8: Transform the list of dictionaries into a dictionary of lists
train_data_dict = defaultdict(list)
for row in train_data_list:
    for key, value in row.items():
        train_data_dict[key].append(value)

test_data_dict = defaultdict(list)
for row in test_data_list:
    for key, value in row.items():
        test_data_dict[key].append(value)

# Convert defaultdict to regular dict for Hugging Face Dataset
train_data_dict = dict(train_data_dict)
test_data_dict = dict(test_data_dict)

# Step 9: Convert to Hugging Face Dataset
hf_train_dataset = Dataset.from_dict(train_data_dict)
hf_test_dataset = Dataset.from_dict(test_data_dict)

# Create DatasetDict
dataset_dict = DatasetDict({
    'train': hf_train_dataset,
    'test': hf_test_dataset
})

# Push to Hugging Face Hub
dataset_dict.push_to_hub("anastasiafrosted/my_sequences_endpoint2_hour")

# Stop Spark session
spark.stop()
