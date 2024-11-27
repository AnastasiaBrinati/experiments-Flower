from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.types import NumericType, ArrayType, DoubleType
from pyspark.sql.functions import col, udf
from pyspark.ml import Pipeline  # Import Pipeline

def scale_csv_data(input_path, output_path, endpoint_id):
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("Scale CSV Data") \
        .getOrCreate()

    # Load the CSV file into a DataFrame
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Identify numeric columns
    numeric_cols = ["invocations_per_hour","avg_loc","avg_cyc_complexity","avg_num_of_imports","avg_argument_size",
                    "avg_total_execution_time","avg_system_processing_time"]

    # Assemble numeric columns into a feature vector
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")

    """
    you can scale features like avg_loc (e.g., 15.0) alongside avg_total_execution_time (e.g., 2.571232×1092.571232×109) 
    as long as you use a scaling method designed to handle the large differences in magnitude effectively:
    Both features are transformed into comparable ranges, eliminating the dominance of large-magnitude features over smaller ones.
    """
    # Scale the features
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

    # Create a Pipeline that includes both steps
    pipeline = Pipeline(stages=[assembler, scaler])

    # Fit the pipeline to the data
    model = pipeline.fit(df)

    # Transform the data using the model
    scaled_df = model.transform(df)

    # Unpack the scaled features vector to individual columns
    unpack_vector_udf = udf(lambda vector: vector.toArray().tolist(), ArrayType(DoubleType()))
    scaled_df = scaled_df.withColumn("scaled", unpack_vector_udf("scaled_features"))

    # Assign unpacked values back to respective columns
    for i, col_name in enumerate(numeric_cols):
        scaled_df = scaled_df.withColumn(col_name, col("scaled").getItem(i))

    # Drop the original vector columns
    scaled_df = scaled_df.drop("features", "scaled_features", "scaled")

    # Save the scaled data to a new CSV file
    scaled_df.coalesce(1).write.csv(output_path, header=True, mode="overwrite")

    # Save the scaler model to a directory
    scaler_model_path = "../helpers/scaler/endpoint"+endpoint_id
    model.stages[1].write().overwrite().save(scaler_model_path)

    # Stop the Spark session
    spark.stop()


if __name__ == "__main__":
    for i in range(10):
        input_csv_path = "globus_data/endpoints/endpoint"+str(i)+"/endpoint"+str(i)+".csv"
        output_csv_path = "scaled_data/globus/endpoints/endpoint"+str(i)+"/"
        scale_csv_data(input_csv_path, output_csv_path, str(i))
