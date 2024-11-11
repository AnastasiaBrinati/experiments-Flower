import pandas as pd
import matplotlib.pyplot as plt

# Load data from the single CSV file
df_original = pd.read_csv("scaled_data/globus/globus_split/train/globus_train.csv")
df = pd.read_csv("results/client1.csv")  # Replace with the actual file path

# Plot the data
plt.figure(figsize=(10, 6))

# Plot execution time
plt.subplot(2, 1, 1)

plt.plot(df_original['timestamp'], df_original['execution_time'], label='Actual Execution Time')
plt.plot(df['timestamps'], df["actual_execution_time"], label="Actual Execution Time", linestyle="-")
plt.plot(df['timestamps'], df["predicted_execution_time"], label="Predicted Execution Time", linestyle="--")
plt.title("Execution Time")
plt.xlabel("Sample Index")
plt.ylabel("Execution Time")
plt.legend()

# Plot cyclomatic complexity
plt.subplot(2, 1, 2)
plt.plot(df['timestamps'], df["actual_cyc_complexity"], label="Actual Cyc Complexity", linestyle="-")
plt.plot(df['timestamps'], df["predicted_cyc_complexity"], label="Predicted Cyc Complexity", linestyle="--")
plt.title("Cyclomatic Complexity")
plt.xlabel("Sample Index")
plt.ylabel("Cyc Complexity")
plt.legend()

plt.tight_layout()
#plt.show()
plt.savefig("client1.png", format="png", dpi=300)  # Save as PNG with high resolution