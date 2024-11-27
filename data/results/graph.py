import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data from the single CSV file
df = pd.read_csv("rescaled_centralized_arima_predictions.csv")  # Replace with the actual file path

# Plot the data
plt.figure(figsize=(18, 6))

# Plot execution time
plt.subplot(2, 1, 1)
plt.plot(df['timestamp'], df["actual_execution_time"], label="Actual Execution Time", linestyle="-", color='blue')
plt.plot(df['timestamp'], df["predicted_execution_time"], label="Predicted Execution Time", linestyle="--", color='orange')
plt.xlabel("Timestamp")
plt.ylabel("Avg Execution Time")
#plt.xticks(rotation=90)  # Rotazione dei timestamp per leggibilità
plt.legend()
plt.grid()

# Plot cyclomatic complexity
plt.subplot(2, 1, 2)
plt.plot(df['timestamp'], df["actual_system_processing_time"], label="Actual System Processing Time", linestyle="-")
plt.plot(df['timestamp'], df["predicted_system_processing_time"], label="Predicted System Processing Time", linestyle="--")
plt.xlabel("Timestamp")
plt.ylabel("Avg System Processing")
#plt.xticks(rotation=90)  # Rotazione dei timestamp per leggibilità
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("endpoint0.png", format="png", dpi=300)  # Save as PNG with high resolution