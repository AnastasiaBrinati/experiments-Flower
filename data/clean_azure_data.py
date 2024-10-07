from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import col, udf
from pyspark.ml.feature import RobustScaler, VectorAssembler, StandardScaler
from pyspark.ml.linalg import DenseVector

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Melt and Filter RDD") \
    .getOrCreate()

# Load the CSV files into DataFrames
dur = spark.read.csv("dur.csv", header=True, inferSchema=True)
# Drop some columns after partitioning
durations = dur.drop('Minimum','Maximum','percentile_Average_0','percentile_Average_1','percentile_Average_25','percentile_Average_50','percentile_Average_75','percentile_Average_99','percentile_Average_100')

mem = spark.read.csv("mem.csv", header=True, inferSchema=True)
# Drop
memory = mem.drop('AverageAllocatedMb_pct1','AverageAllocatedMb_pct5','AverageAllocatedMb_pct25','AverageAllocatedMb_pct50','AverageAllocatedMb_pct75','AverageAllocatedMb_pct95','AverageAllocatedMb_pct99','AverageAllocatedMb_pct100')

func = spark.read.csv("func.csv", header=True, inferSchema=True)

# Specify the columns on which to join the DataFrames
join_columns = ['HashOwner', 'HashApp']  # Columns common to both DataFrames

# Perform an inner join on the two DataFrames based on the specified columns
merged_df = durations.join(memory, on=join_columns, how='inner')
# Perform an inner join on the two DataFrames based on the specified columns
merged_df = merged_df.join(func, on=join_columns, how='inner')

# Specify the ID columns and assume the rest are minute columns
id_columns = ['HashOwner', 'HashApp', 'HashFunction', 'Trigger', 'Average','Count','SampleCount','AverageAllocatedMb']
minute_columns = [col for col in merged_df.columns if col not in id_columns]

# Convert the DataFrame to an RDD
rdd = merged_df.rdd

unique_triggers = rdd.map(lambda row: row['Trigger']).distinct().collect()
# Use flatMap to melt the DataFrame, transforming it from wide to long format
melted_rdd = rdd.flatMap(lambda row: [
    Row(
        HashOwner=row['HashOwner'],
        HashApp=row['HashApp'],
        HashFunction=row['HashFunction'],
        AverageDuration=row['Average'],
        CountForDuration=row['Count'],
        AverageAllocatedMb=row['AverageAllocatedMb'],
        CountForMemory=row['SampleCount'],
        minute=minute,
        invocations=getattr(row, minute),
        **{f'Trigger_{trigger}': 1 if row['Trigger'] == trigger else 0 for trigger in unique_triggers}  # One-hot encoding for all unique triggers
    ) for minute in minute_columns if getattr(row, minute) != 0  # Filter out zero invocations
])


# Convert the transformed RDD back to a DataFrame
filtered_df = spark.createDataFrame(melted_rdd)

filtered_df.show(10)

# delete later
partitioned_df = filtered_df.drop('HashOwner', 'HashApp')
partitioned_df.write.csv("total/", header=True)

'''
# Scaling
# Step 1: Assemble the numeric columns into a vector for scaling
assembler = VectorAssembler(
    inputCols=['AverageDuration', 'CountForDuration', 'AverageAllocatedMb', 'CountForMemory', 'invocations'],
    outputCol='features'
)
assembled_df = assembler.transform(filtered_df)

# Step 2: Apply the StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
scaler_model = scaler.fit(assembled_df)
scaled_df = scaler_model.transform(assembled_df)

# Step 3: Define a UDF to extract scaled values from the DenseVector
def extract_scaled_values(vector):
    return tuple(vector.toArray())

extract_udf = udf(extract_scaled_values)

# Step 4: Apply UDF to create new scaled columns and drop the original columns
scaled_df = scaled_df.withColumn("scaled_values", extract_udf(col("scaledFeatures")))

# Split the scaled values into separate columns
final_df = scaled_df.select(
    col("scaled_values").getItem(0).alias("scaled_AverageDuration"),
    col("scaled_values").getItem(1).alias("scaled_CountForDuration"),
    col("scaled_values").getItem(2).alias("scaled_AverageAllocatedMb"),
    col("scaled_values").getItem(3).alias("scaled_CountForMemory"),
    col("scaled_values").getItem(4).alias("scaled_invocations"),
    col('HashOwner'),  # Retain non-scaled columns
    col('HashApp'),
    col('HashFunction'),
    col('Trigger')
)

final_df.show(10)



# Repartition the DataFrame (e.g., into 5 partitions) by 'HashOwner'
number_of_partitions = 5
partitioned_df = filtered_df.repartition(number_of_partitions, "HashOwner")

# Drop the 'HashOwner' column after partitioning
partitioned_df = partitioned_df.drop('HashOwner', 'HashApp')

# Show the partitioned and filtered data (first 10 rows)
partitioned_df.show(10)

# Save the partitioned DataFrame to CSV
partitioned_df.write.csv("partitioned_data/", header=True)
'''



# Stop Spark session
spark.stop()
