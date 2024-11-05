from pyspark.sql import SparkSession

def split_csv_data(input_path, output_path):
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("Split CSV Data") \
        .getOrCreate()

    # Load the CSV file into a DataFrame
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Split into train and test set
    df_train, df_test = df.randomSplit([0.5, 0.5], seed=42)

    # Split the data into two DataFrames (30% and 70%)
    train_30, train_70 = df_train.randomSplit([0.3, 0.7], seed=42)
    test_30, test_70 = df_test.randomSplit([0.3, 0.7], seed=42)

    # This is for Globus
    # train_30, train_70 = train_30.randomSplit([0.3, 0.7], seed=42)
    # test_30, test_70 = test_30.randomSplit([0.3, 0.7], seed=42)

    # Save each split DataFrame to a separate CSV file
    train_30.coalesce(1).write.csv(output_path+"/train", header=True, mode="overwrite")
    test_30.coalesce(1).write.csv(output_path+"/test", header=True, mode="overwrite")

    # Stop the Spark session
    spark.stop()


# Example usage
input_csv_path = "scaled_data/globus/globus_scaled.csv"
output_csv_path = "scaled_data/globus/globus_30"
split_csv_data(input_csv_path, output_csv_path)
