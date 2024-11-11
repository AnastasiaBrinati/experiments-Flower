from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max, date_sub
from datetime import timedelta

# SPLITTING the time series 'randomly', as the test set is the last 2 weeks

def split_csv_data(input_path, output_path):
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("Split CSV Data") \
        .getOrCreate()

    # Load the CSV file into a DataFrame
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Ensure the timestamp column is in datetime format
    # If it's not, you can convert it using:
    # df = df.withColumn("timestamp", col("timestamp").cast("timestamp"))

    # Get the maximum timestamp in the dataset
    max_date = df.agg(max(col("date"))).collect()[0][0]

    # Calculate the cutoff date for the last 2 weeks
    cutoff_date = max_date - timedelta(days=21)

    # Split the data into training and testing sets based on the timestamp
    df_train = df.filter(col("date") < cutoff_date)
    df_test = df.filter(col("date") >= cutoff_date)

    # NO NEED in globus becuase smaller

    # Split into train and test set
    # df_train, df_test = df.randomSplit([0.5, 0.5], seed=42)

    # Split the data into two DataFrames (30% and 70%)
    # train_30, train_70 = df_train.randomSplit([0.3, 0.7], seed=42)
    # test_30, test_70 = df_test.randomSplit([0.3, 0.7], seed=42)

    # This is for Globus
    # train_30, train_70 = train_30.randomSplit([0.3, 0.7], seed=42)
    # test_30, test_70 = test_30.randomSplit([0.3, 0.7], seed=42)

    # Save each split DataFrame to a separate CSV file
    df_train.coalesce(1).write.csv(output_path+"/train", header=True, mode="overwrite")
    df_test.coalesce(1).write.csv(output_path+"/test", header=True, mode="overwrite")

    # Stop the Spark session
    spark.stop()


# Example usage
input_csv_path = "scaled_data/globus/globus_scaled.csv"
output_csv_path = "scaled_data/globus/globus_split"
split_csv_data(input_csv_path, output_csv_path)
