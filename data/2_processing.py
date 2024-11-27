from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import isnan
from pyspark.sql.types import LongType
from pyspark.sql.functions import from_unixtime, date_format, hour, minute, second, udf
from pyspark.sql.functions import col, count, avg, expr, concat_ws, first
from pyspark.sql import functions as F

# Initialize Spark session
spark = SparkSession.builder.appName("Melt and Filter Globus dataset").getOrCreate()

# Load the CSV file into DataFrame
data = spark.read.csv("globus_data/total/globus.csv", header=True, inferSchema=True)

# Drop unnecessary columns
data = data.drop("task_uuid")
data = data.drop("function_uuid")
version = [col for col in data.columns if col.startswith("e_vers")]
data = data.drop(*version)

# Formulate target columns
# 1: we already have total_execution_time
# 2: we want from 'waiting for nodes' to 'execution end' = scheduling_time + queue_time + execution_time
data = data.withColumn("system_processing_time", (col("scheduling_time") + col("queue_time") + col("execution_time")))


# Select columns relative to endpoint types
endpoint_type_columns = [c for c in data.columns if c.startswith("e_type_")]
# Grouping and calculations
result = (
    data.groupBy("endpoint_uuid", "timestamp")
    .agg(
        count("*").alias("invocations_per_hour"),  # Numero di invocazioni
        avg("loc").alias("avg_loc"),  # Media delle linee di codice
        avg("cyc_complexity").alias("avg_cyc_complexity"),  # Media della complessità ciclica
        avg("num_of_imports").alias("avg_num_of_imports"),  # Media del numero di importazioni
        avg("argument_size").alias("avg_argument_size"),  # Media della dimensione degli argomenti
        *[first(col(c)).alias(c) for c in endpoint_type_columns],  # Mantenimento delle colonne `e_type_*`
        avg("total_execution_time").alias("avg_total_execution_time"),  # Media del tempo di esecuzione
        avg("system_processing_time").alias("avg_system_processing_time")  # Media del tempo di scheduling
    )
)

# more Ordering
result = result.orderBy("timestamp")

# Save in one file
result.coalesce(1).write.csv("globus_data/flipped/", header=True, mode="overwrite")

# Stop Spark session
spark.stop()