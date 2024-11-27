from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max, date_sub
from datetime import timedelta
from datetime import datetime
from pyspark.sql.functions import to_date, regexp_replace, split, concat_ws, count, desc

def max_endpoints(input_path, output_path):
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("Split CSV Data") \
        .getOrCreate()

    # Load the CSV file into a DataFrame
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Step 1: Count the frequency of each endpoint_uuid
    endpoint_frequency = (
        df.groupBy("endpoint_uuid")
        .agg(count("*").alias("frequency"))
        .orderBy(desc("frequency"))
    )

    # How many endpoints?
    limit = 10

    # Step 2: Get the top 2 most frequent endpoints
    top_endpoints = endpoint_frequency.limit(limit).select("endpoint_uuid").rdd.flatMap(lambda x: x).collect()

    for i in range(limit):
        # Step 3: Partition the data for the top 2 endpoints
        partition = df.filter(col("endpoint_uuid") == top_endpoints[i])
        # Step 4: Save the partitions to separate CSV files
        partition.coalesce(1).write.csv(output_path + "/endpoint"+str(i), header=True, mode="overwrite")

    # Stop the Spark session
    spark.stop()


# Example usage
input_csv_path = "globus_data/flipped/globus.csv"
output_csv_path = "globus_data/endpoints/"
max_endpoints(input_csv_path, output_csv_path)
