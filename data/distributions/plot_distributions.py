import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from the single CSV file
df = pd.read_csv("../globus_data/endpoints/endpoint0/endpoint0.csv")  # Replace with the actual file path

# Convert scientific notation to float for better plotting
df['avg_total_execution_time'] = df['avg_total_execution_time'].astype(float)
df['avg_system_processing_time'] = df['avg_system_processing_time'].astype(float)

# List of numeric columns to plot
numeric_columns = [
    'invocations_per_hour',
    'avg_loc',
    'avg_cyc_complexity',
    'avg_argument_size',
    'avg_total_execution_time',
    'avg_system_processing_time'
]

# Plot the distribution of each numeric column
for column in numeric_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=True, bins=10, color='blue', alpha=0.7)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(column+"_distribution.png", format="png", dpi=300)  # Save as PNG with high resolution
