from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max, date_sub
from datetime import timedelta
from datetime import datetime
from pyspark.sql.functions import to_date, regexp_replace, split, concat_ws

# SPLITTING the time series 'randomly', as the test set is the last 2 weeks

def split_csv_data(input_path, output_path):
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("Split CSV Data") \
        .getOrCreate()

    # Load the CSV file into a DataFrame
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Split the timestamp column by ':' and recombine it with a space
    df = df.withColumn("formatted_timestamp", concat_ws(" ", split(col("timestamp"), ":", 2)))
    # Convert the resulting string into a proper timestamp type
    df = df.withColumn("formatted_timestamp", col("formatted_timestamp").cast("timestamp"))

    df = df.withColumn("date", to_date(col("formatted_timestamp"), "yyyy-MM-dd"))

    # Get the maximum value of the 'date' column
    max_date = df.agg(max(col("date"))).collect()[0][0]  # Retrieve the max date
    print(f"Maximum date: {max_date}")

    # Calculate the cutoff date
    cutoff_date = max_date - timedelta(days=21)
    print(f"Cutoff date: {cutoff_date}")

    # Split the data into training and testing sets based on the timestamp
    df_train = df.filter(col("date") < cutoff_date)
    df_test = df.filter(col("date") >= cutoff_date)

    df_train = df_train.drop("date")
    df_train = df_train.drop("formatted_timestamp")
    df_test = df_test.drop("date")
    df_test = df_test.drop("formatted_timestamp")

    # Save each split DataFrame to a separate CSV file
    df_train.coalesce(1).write.csv(output_path+"/train", header=True, mode="overwrite")
    df_test.coalesce(1).write.csv(output_path+"/test", header=True, mode="overwrite")

    # Stop the Spark session
    spark.stop()


# Example usage
input_csv_path = "scaled_data/globus/endpoints/endpoint2/scaled_endpoint_2.csv"
output_csv_path = "scaled_data/globus/endpoints/endpoint2/split"
split_csv_data(input_csv_path, output_csv_path)
