import pandas as pd

def interpolate(input_csv_path, output_csv_file, endpoint_id):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)  # Assuming the first column is a datetime index

    # Ensure the index is datetime-based for asfreq and interpolation
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame's index must be a DatetimeIndex.")

    # Adjust frequency to seconds and interpolate missing values
    df = df.asfreq('s').interpolate(method='time')

    # Save the processed DataFrame to a new CSV file
    df.to_csv(output_path)
    print(f"Processed CSV saved to {output_path}")

if __name__ == "__main__":
    for i in range(10):
        input_csv_path = "scaled_data/globus/endpoints/endpoint"+str(i)+"/endpoint"+str(i)+".csv"
        output_csv_path = "scaled_data/globus/interpolated/endpoints/endpoint"+str(i)+"/"
        interpolate(input_csv_path, output_csv_path, str(i))
