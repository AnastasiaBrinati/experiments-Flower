import pandas as pd
import ast
import ast
import numpy as np

# Read the CSV file
df = pd.read_csv("../client_1_predictions.csv")

# Select the first row
timestamps = df.iloc[0][0]

# Step 1: Replace things
timestamps = timestamps.replace("'", "\"")
timestamps = timestamps.replace("]", "],")
timestamps = timestamps.replace("],],", "]],")
timestamps = timestamps.replace("]],],", "]]]")
timestamps = timestamps.replace("\" \"", "\",\"")
timestamps = timestamps.replace("...", "")
timestamps = ast.literal_eval(timestamps)

#print(timestamps)
flattened_timestamps = [item for sublist in timestamps for item in sublist]
time = pd.DataFrame(flattened_timestamps, columns=['timestamps', 'date'])
time = time['timestamps']
#time = pd.DataFrame({"timestamp": timestamps})
#print(time)

actual_execution_time = df.iloc[0][1]
# Step 1: Replace things
actual_execution_time = actual_execution_time.replace(" ", ",")
actual_execution_time = actual_execution_time.replace(",...", "")
actual_execution_time = actual_execution_time.replace(",,", ",")
actual_execution_time = ast.literal_eval(actual_execution_time)
act_exe = pd.DataFrame({"actual_execution_time": actual_execution_time})
act_exe = act_exe.explode("actual_execution_time").reset_index(drop=True)


predicted_execution_time = df.iloc[0][2]
# Step 1: Replace things
predicted_execution_time = predicted_execution_time.replace(" ", ",")
predicted_execution_time = predicted_execution_time.replace(",,", ",")
predicted_execution_time = predicted_execution_time.replace(",...", "")
predicted_execution_time = ast.literal_eval(predicted_execution_time)
pred_exe = pd.DataFrame({"predicted_execution_time": predicted_execution_time})
pred_exe = pred_exe.explode("predicted_execution_time").reset_index(drop=True)

actual_cc = df.iloc[0][3]
# Step 1: Replace things
actual_cc = actual_cc.replace(" ", ",")
actual_cc = actual_cc.replace(",...", "")
actual_cc = actual_cc.replace(",,,", ",")
actual_cc = actual_cc.replace(",,", ",")
actual_cc = actual_cc.replace(",]", "]")
actual_cc = ast.literal_eval(actual_cc)
act_cc = pd.DataFrame({"actual_cyc_complexity": actual_cc})
act_cc = act_cc.explode("actual_cyc_complexity").reset_index(drop=True)

predicted_cc = df.iloc[0][4]
# Step 1: Replace things
predicted_cc = predicted_cc.replace(" ", ",")
predicted_cc = predicted_cc.replace(",,", ",")
predicted_cc = predicted_cc.replace(",...", "")
predicted_cc = ast.literal_eval(predicted_cc)
pred_cc = pd.DataFrame({"predicted_cyc_complexity": predicted_cc})
pred_cc = pred_cc.explode("predicted_cyc_complexity").reset_index(drop=True)

data = pd.concat([time, act_exe], axis=1)
data = pd.concat([data, pred_exe], axis=1)
data = pd.concat([data, act_cc], axis=1)
data = pd.concat([data, pred_cc], axis=1)

data = data.sort_values(by=["timestamps"])
print(data)
data.to_csv("results/client1.csv", index=False)