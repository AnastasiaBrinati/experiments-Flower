from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import isnan
from pyspark.sql.types import LongType
from pyspark.sql.functions import from_unixtime, date_format, hour, minute, second, udf
from pyspark.sql.functions import col, count, avg, expr, concat_ws, first
from pyspark.sql import functions as F

# Initialize Spark session
spark = SparkSession.builder.appName("Melt and Filter Globus dataset").getOrCreate()

# Load the CSV files into DataFrames
# ENDPOINTS
endpoints = spark.read.csv("globus_data/endpoints.csv", header=True, inferSchema=True)
# FUNCTIONS
functions = spark.read.csv("globus_data/functions.csv", header=True, inferSchema=True).drop('function_body_uuid')
# TASKS
tasks = spark.read.csv("globus_data/tasks.csv", header=True, inferSchema=True).drop('anonymized_user_uuid')

# Perform an inner join on the two DataFrames based on the specified columns
globus = tasks.join(endpoints, on=['endpoint_uuid'], how='inner')
# Perform an inner join on the two DataFrames based on the specified columns
globus = globus.join(functions, on=['function_uuid'], how='inner').dropna(how='any')

globus = globus.withColumn("date", date_format(from_unixtime((col("received") / 1_000_000_000).cast(LongType())), "yyyy-MM-dd"))
globus = globus.withColumn("hour", hour(from_unixtime((col("received") / 1_000_000_000).cast(LongType()))))
globus = globus.withColumn("minute", minute(from_unixtime((col("received") / 1_000_000_000).cast(LongType()))))

# Creazione della colonna 'minute'
df = globus.withColumn("timestamp", concat_ws(":", col("date"), col("hour"), col("minute")))
# Calcolo della differenza timestamps in  nuove colonne
df = df.withColumn("execution_time", (col("execution_end") - col("execution_start")))
df = df.withColumn("scheduling_time", (col("execution_start") - col("received")))

# Step 1: Get unique `endpoint_type` values
unique_endpoint_types = [row['endpoint_type'] for row in df.select("endpoint_type").distinct().collect()]

# Step 2: Dynamically add columns for each endpoint type
for e_type in unique_endpoint_types:
    column_name = f"e_type_{e_type.replace('.', '_')}"  # Replace dots with underscores
    df = df.withColumn(column_name, F.when(F.col("endpoint_type") == e_type, 1).otherwise(0))

    # Selezione delle colonne relative a endpoint types
    endpoint_type_columns = [c for c in df.columns if c.startswith("e_type_")]

# Raggruppamento e calcolo delle metriche
result = (
    df.groupBy("endpoint_uuid", "timestamp")
    .agg(
        count("*").alias("invocations_per_minute"),  # Numero di invocazioni
        avg("loc").alias("avg_loc"),  # Media delle linee di codice
        avg("cyc_complexity").alias("avg_cyc_complexity"),  # Media della complessità ciclica
        avg("num_of_imports").alias("avg_num_of_imports"),  # Media del numero di importazioni
        avg("argument_size").alias("avg_argument_size"),  # Media della dimensione degli argomenti
        *[first(col(c)).alias(c) for c in endpoint_type_columns],  # Mantenimento delle colonne `e_type_*`
        avg("execution_time").alias("avg_execution_time"),  # Media del tempo di esecuzione
        avg("scheduling_time").alias("avg_scheduling_time")  # Media del tempo di scheduling
    )
)

# Ordinare il risultato
result = result.orderBy("timestamp")

result.coalesce(1).write.csv("globus_data/flipped/", header=True, mode="overwrite")

# Stop Spark session
spark.stop()