from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import col, udf
from pyspark.ml.feature import RobustScaler, VectorAssembler, StandardScaler
from pyspark.ml.linalg import DenseVector
from pyspark.sql.functions import isnan

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Melt and Filter RDD") \
    .getOrCreate()

# Load the CSV files into DataFrames
# ENDPOINTS
endpoints = spark.read.csv("globus_data/endpoints.csv", header=True, inferSchema=True)

# FUNCTIONS
functions = spark.read.csv("globus_data/functions.csv", header=True, inferSchema=True)
# Drop
functions = functions.drop('function_body_uuid')

# TASKS
tasks = spark.read.csv("globus_data/tasks.csv", header=True, inferSchema=True)
# Drop
tasks = tasks.drop('anonymized_user_uuid')


# Perform an inner join on the two DataFrames based on the specified columns
df_joined = tasks.join(endpoints, on=['endpoint_uuid'], how='inner')
# Perform an inner join on the two DataFrames based on the specified columns
tasks_joined = df_joined.join(functions, on=['function_uuid'], how='inner')

# Drop NaN values
df_clean = tasks_joined.dropna(how='any')
df_clean.show(10)

# Convert the DataFrame to an RDD
rdd = df_clean.rdd

unique_endpoint_type = rdd.map(lambda row: row['endpoint_type']).distinct().collect()
unique_endpoint_version = rdd.map(lambda row: row['endpoint_version']).distinct().collect()

# Use flatMap to melt the DataFrame, transforming it from wide to long format
melted_rdd = rdd.flatMap(lambda row: [
    Row(
        # from tasks
        function_uuid=row['function_uuid'],
        endpoint_uuid=row['endpoint_uuid'],
        task_uuid=row['task_uuid'],
        argument_size=row['argument_size'],
        assignment_time=row['waiting_for_nodes'] - row['received'], # time between the task arriving to the cloud platform the task being received by an endpoint
        scheduling_time=row['waiting_for_launch'] - row['waiting_for_nodes'], # time between the task received by an endpoint and the task assigned to a worker
        queue_time=row['execution_start'] - row['waiting_for_launch'], # time between the task assigned to a worker and the execution start
        execution_time=row['execution_end'] - row['execution_start'], # time between the task execution start and end
        total_execution_time=row['result_received'] - row['received'], # time between the task arriving and the results being reported, to the could platform

        # from functions
        loc=row['loc'],
        cyc_complexity=row['cyc_complexity'],
        num_of_imports=row['num_of_imports'],

        # from endpoints
        **{f"e_type_{e_type.replace('.', '_')}": 1 if row['endpoint_type'] == e_type else 0 for e_type in unique_endpoint_type},  # One-hot encoding for all unique endpoint types
        **{f"e_vers_{e_vers.replace('.', '_')}": 1 if row['endpoint_version'] == e_vers else 0 for e_vers in unique_endpoint_version}  # One-hot encoding for all unique endpoint versions
    )
])

# Convert the transformed RDD back to a DataFrame
filtered_df = spark.createDataFrame(melted_rdd)
#filtered_df = filtered_df.drop('endpoint_uuid', 'task_uuid')
filtered_df = filtered_df.dropna(how='any')

# Shuffle
shuffled_df = filtered_df.sample(withReplacement=False, fraction=1.0, seed=42).repartition(filtered_df.rdd.getNumPartitions())

# Repartition the DataFrame
df_repartitioned = shuffled_df.repartition(1)
# Save the DataFrame in CSV format
df_repartitioned.write.format("csv").option("header", "true").save("globus_data/total")


# Stop Spark session
spark.stop()
