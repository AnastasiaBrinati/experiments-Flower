from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import row_number, collect_list, col, size, slice

# Initialize Spark session
spark = SparkSession.builder.appName("LSTMSequencePreparation").getOrCreate()

# Load data and assume the CSV file includes an 'endpoint_uuid', 'date', 'feature1', 'feature2', and 'target' columns
train_data = spark.read.csv("scaled_data/globus/globus_split/train/globus_train.csv", header=True, inferSchema=True)
#test_data = spark.read.csv("scaled_data/globus/globus_split/test/globus_test.csv", header=True, inferSchema=True)

# Step 0: Sequence and prediction lengths
sequence_length = 7
prediction_length = 3
total_length = sequence_length + prediction_length

# Step 1: Partition by 'id' and order by 'date'
window = Window.orderBy("timestamp")

# Step 2: Create a row number within each partition, ensuring sequential order within each 'id'
train_data = train_data.withColumn("row_num", row_number().over(window))
#test_data = test_data.withColumn("row_num", row_number().over(window))

# Define input and output features
input_features = ["timestamp", "invocations_per_minute"
                  "avg_argument_size", "avg_loc", "avg_cyc_complexity", "avg_num_of_imports",
                  "e_type_LSFProvider","e_type_CobaltProvider","e_type_PBSProProvider",
                  "e_type_LocalProvider","e_type_KubernetesProvider","e_type_SlurmProvider"]
output_features = ["avg_execution_time", "avg_scheduling_time"]

# Step 3: Create sliding window sequences for input and target features only
sliding_window = Window.orderBy("row_num").rowsBetween(0, total_length - 1)

# Step 4: Collect input feature sequences
for feature in input_features + output_features:
    train_data = train_data.withColumn(f"{feature}_seq", collect_list(col(feature)).over(sliding_window))
    #test_data = test_data.withColumn(f"{feature}_seq", collect_list(col(feature)).over(sliding_window))

# Step 5: Filter for full sequences
train_data = train_data.filter(size(col(f"{input_features[0]}_seq")) == total_length)
#test_data = test_data.filter(size(col(f"{input_features[0]}_seq")) == total_length)

# Step 6: Extract input and target sequences using slice
for feature in input_features:
    train_data = train_data.withColumn(f"{feature}_seq_input", slice(col(f"{feature}_seq"), 1, sequence_length))
    #test_data = test_data.withColumn(f"{feature}_seq_input", slice(col(f"{feature}_seq"), 1, sequence_length))

for feature in output_features:
    train_data = train_data.withColumn(f"{feature}_seq_target", slice(col(f"{feature}_seq"), sequence_length + 1, prediction_length))
    #test_data = test_data.withColumn(f"{feature}_seq_target",
    #                                   slice(col(f"{feature}_seq"), sequence_length + 1, prediction_length))

# Step 7: Select only the required columns for input and target
train_data = train_data.select(
    *[col(f"{feature}_seq_input").alias(f"{feature}_seq") for feature in input_features],
    *[col(f"{feature}_seq_target").alias(f"{feature}_target") for feature in output_features]
)

#test_data = test_data.select(
#    *[col(f"{feature}_seq_input").alias(f"{feature}_seq") for feature in input_features],
#    *[col(f"{feature}_seq_target").alias(f"{feature}_target") for feature in output_features]
#)

# Step 8: Convert Spark df in Python dict
train_data_dict = train_data.rdd.map(lambda row: row.asDict()).collect()
# test_data_dict = test_data.rdd.map(lambda row: row.asDict()).collect()

from datasets import Dataset, DatasetDict

# Step 9: Convert to Hugging Face Dataset
hf_train_dataset = Dataset.from_dict(train_data_dict)
# hf_test_dataset = Dataset.from_dict(test_data_dict)

dataset_dict = DatasetDict({
    'train': hf_train_dataset,
    #'test': hf_test_dataset
})

# Step 10 eventual: Save the DatasetDict to a local directory
#dataset_dict.save_to_disk("my_sequences_dataset")
dataset_dict.push_to_hub("anastasiafrosted/my_sequences_dataset")


# Stop Spark session
spark.stop()
