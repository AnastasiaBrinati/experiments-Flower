from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max, date_sub, to_timestamp
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

    # Convert the 'timestamp' column to a proper timestamp format
    df = df.withColumn("formatted_timestamp", to_timestamp(col("timestamp"), "yyyy-MM-dd'T'HH:mm:ss.SSSXXX"))

    # Extract the date from the timestamp
    df = df.withColumn("date", to_date(col("formatted_timestamp")))

    # Get the maximum value of the 'date' column
    max_date = df.agg(max(col("date"))).collect()[0][0]  # Retrieve the max date
    print(f"Maximum date: {max_date}")

    # Calculate the cutoff date
    cutoff_date = max_date - timedelta(days=7)
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

if __name__ == "__main__":
    for i in range(10):
        input_csv_path = "scaled_data/globus/endpoints/endpoint"+str(i)+"/endpoint"+str(i)+".csv"
        output_csv_path = "scaled_data/globus/endpoints/endpoint"+str(i)+"/split"
        split_csv_data(input_csv_path, output_csv_path)
