from pyspark.sql import SparkSession

def split_csv_data(train_input_path, test_input_path, train_output_path_30, test_output_path_30):
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("Split CSV Data") \
        .getOrCreate()

    # Load the CSV file into a DataFrame
    df_train = spark.read.csv(train_input_path, header=True, inferSchema=True)
    df_test = spark.read.csv(test_input_path, header=True, inferSchema=True)

    shuffled_train = df_train.sample(withReplacement=False, fraction=1.0, seed=42).repartition(df_train.rdd.getNumPartitions())
    shuffled_test = df_test.sample(withReplacement=False, fraction=1.0, seed=42).repartition(df_test.rdd.getNumPartitions())

    # Split the data into two DataFrames (30% and 70%)
    train_30, train_70 = shuffled_train.randomSplit([0.3, 0.7], seed=42)
    train_30, train_70 = train_30.randomSplit([0.3, 0.7], seed=42)
    test_30, test_70 = shuffled_test.randomSplit([0.3, 0.7], seed=42)
    test_30, test_70 = test_30.randomSplit([0.3, 0.7], seed=42)

    # Save each split DataFrame to a separate CSV file
    train_30.coalesce(1).write.csv(train_output_path_30, header=True, mode="overwrite")
    test_30.coalesce(1).write.csv(test_output_path_30, header=True, mode="overwrite")

    # Stop the Spark session
    spark.stop()

# Example usage
train_input_csv_path = "scaled_data/func/scaled_train.csv"
test_input_csv_path = "scaled_data/func/scaled_test.csv"
train_output_csv_path_30 = "scaled_data/func/func_train_30"
test_output_csv_path_30 = "scaled_data/func/func_test_30"
split_csv_data(train_input_csv_path, test_input_csv_path, train_output_csv_path_30, test_output_csv_path_30)
