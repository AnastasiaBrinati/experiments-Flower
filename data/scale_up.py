from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.feature import StandardScalerModel

def reverse_scaling(input_path, scaler_model_path, feature_cols):
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("Reverse Scaling") \
        .getOrCreate()

    # Load the scaled data
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Load the saved StandardScalerModel
    scaler_model = StandardScalerModel.load(scaler_model_path)

    # Retrieve the standard deviation and mean vectors from the scaler model
    std_dev = scaler_model.std.toArray()
    mean = scaler_model.mean.toArray()  # Use only if mean centering was enabled

    # Reverse scaling for each column:
    # avg_total_execution_time was numeric column number 6 !!!!!
    i = 5
    print(f"Reversing scaling for column: {features_to_inverse[0]}")
    df = df.withColumn(features_to_inverse[0], col(features_to_inverse[0]) * std_dev[i] + mean[i]) #Reversing
    print(f"Reversing scaling for column: {features_to_inverse[1]}")
    df = df.withColumn(features_to_inverse[1], col(features_to_inverse[1]) * std_dev[i] + mean[i]) #Reversing

    # avg_system_processing_time was numeric column number 7 !!!!!
    j = 6
    print(f"Reversing scaling for column: {features_to_inverse[2]}")
    df = df.withColumn(features_to_inverse[2], col(features_to_inverse[2]) * std_dev[j] + mean[j]) #Reversing
    print(f"Reversing scaling for column: {features_to_inverse[3]}")
    df = df.withColumn(features_to_inverse[3], col(features_to_inverse[3]) * std_dev[j] + mean[j]) #Reversing


    # Save the reversed data to a CSV
    df.coalesce(1).write.csv("reverse_scaling", header=True, mode="overwrite")

    # Stop the Spark session
    spark.stop()


if __name__ == "__main__":
    scaled_csv_path = "centralized_arima_predictions.csv"
    scaler_model_path = "../helpers/scaler/endpoint0"

    features_to_inverse = [
        "actual_execution_time",
        "predicted_execution_time",
        "actual_system_processing_time",
        "predicted_system_processing_time"
    ]

    # Step 2: Reverse the scaling
    reverse_scaling(scaled_csv_path, scaler_model_path, features_to_inverse)