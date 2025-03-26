import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

path_to_folder = r"C:\Users\clara\Desktop\special_course\ds003838"

subnums = np.arange(32, 99, 1) # shouldn't it be to 99?
subnums = np.delete(subnums, np.where((subnums == 37) | (subnums == 66))) #shouldnt we remove 94 too?
subject_folders = [f'sub-0{i}' for i in subnums]

# Initialize an empty list to store DataFrames
all_subjects_events = []

for i in subject_folders:
    folder_path = os.path.join(path_to_folder, i, "pupil")
    filename = i + "_task-memory_events.tsv"
    file_path = os.path.join(folder_path, filename)

    # Read each CSV file into a DataFrame
    try:
        df = pd.read_csv(file_path, sep='\t', usecols=['timestamp', 'label'])
        all_subjects_events.append(df)
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")

    except:
        print(file_path)

# for i in all_subjects_events:
#     print(i)

# Load the TSV file
# file_path = r"C:\Users\clara\Desktop\special_course\ds003838\sub-033\pupil\sub-033_task-memory_events.tsv"
# df = pd.read_csv(file_path, sep="\t")  # Read the TSV file

# First number is the number of recalled items, second number is the total number of items
# Initialize a dictionary to store the number of recalled and total items for each category
category_map = {
    "05": np.zeros((5, 2), dtype=int),  # Shape (5,2) for recalled & total
    "09": np.zeros((9, 2), dtype=int),
    "13": np.zeros((13, 2), dtype=int),
}

print(f"Number of subjects: {len(all_subjects_events)}")
for idx, df in enumerate(all_subjects_events):
    if df.shape[0] != 1458:
        print(f"Subject {subject_folders[idx]}: {df.shape}")
# print(df.head())

# for label in df["label"].astype(str):  # Convert entire column to string at once
for i in range(0, len(all_subjects_events)):
    for label in all_subjects_events[i]["label"].astype(str):  # Convert entire column to string at once
        if label.startswith("60"):  # Ensure valid label
            load_position = int(label[2:4]) - 1  # 3rd and 4th digits (zero-indexed)
            category_key = label[4:6]  # 5th and 6th digits (05, 09, 13)
            recall_flag = int(label[-1])  # Last digit (0 or 1)

            if category_key in category_map:
                category_map[category_key][load_position, 1] += 1  # Increment total count
                category_map[category_key][load_position, 0] += recall_flag  # Increment recalled count if recall_flag is 1

# Print results
for key, array in category_map.items():
    print(f"Category {key}:", array.tolist())  # Convert NumPy array to list for readability


# Compute means using NumPy
means = {key: np.divide(array[:, 0], array[:, 1], where=array[:, 1] > 0) for key, array in category_map.items()}

# Create x-axis values
x_values = {key: np.arange(1, len(means[key]) + 1) for key in means}

# Plot the means for each category
plt.figure(figsize=(8, 6))
plt.plot(x_values["05"], means["05"], marker='o', linestyle='-', label='Category 05')
plt.plot(x_values["09"], means["09"], marker='s', linestyle='-', label='Category 09')
plt.plot(x_values["13"], means["13"], marker='^', linestyle='-', label='Category 13')

# Labels and title
plt.xlabel('Load Position')
plt.ylabel('Mean Recall Rate')
plt.title('Mean Recall Rate per Load Position')
plt.legend()
plt.grid(True)

# Show plot
plt.show()