# %% [markdown]
# <h3> Pupillometry Preprocessing </h3>

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_to_folder = r"C:\Users\Bruger\Documents\DTU\Thesis\Data\ds003838"

subnums = np.arange(32, 100, 1) # shouldn't it be to 99?
subnums = np.delete(subnums, np.where((subnums == 37) | (subnums == 66))) #shouldnt we remove 94 too?
subject_folders = [f'sub-0{i}' for i in subnums]


# %% [markdown]
# Loading all the event code data:

# %%
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

for i in all_subjects_events:
    print(i)

print(len(all_subjects_events))

# %% [markdown]
# Loading all the pupillometry data:

# %%
# Initialize an empty list to store DataFrames
all_subjects_pupil = []

for i in subject_folders:
    folder_path = os.path.join(path_to_folder, i, "pupil")
    filename = i + "_task-memory_pupil.tsv"
    file_path = os.path.join(folder_path, filename)

    
    # Read each CSV file into a DataFrame
    try:
        df = pd.read_csv(file_path, sep='\t', usecols=['pupil_timestamp', 'diameter', 'confidence', 'blink'])
        all_subjects_pupil.append(df)
        print(f"File loaded: {file_path}")
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")

    except Error as e:
        print(f"{e} loading {file_path}")

print(len(all_subjects_pupil))

# %% [markdown]
# Adding missing values to the diameter column where the confidence is lower than 80%:

# %%
print(len(all_subjects_pupil))

for i in all_subjects_pupil:
    # Conditions: 'confidence' < 0.8 or 'blink' == 1
    condition = i['confidence'] < 0.8

    # Replace values with NaN for rows that meet the condition
    i.loc[condition, 'diameter'] = np.nan

    print(i.head(10))

# %%
control5 = []

for i in all_subjects_events:
    # labels for memorize 1st digist in 5, 9 or 13 digit sequence (recolled or forgotte)
    control5.append(i[i['label'].isin([6001131, 6001130, 6001050, 6001051, 6001090, 6001091])]) 
    # labelos for control, listen to 1st digit in 5, 9 or 13 digit sequence
    # control5.append(i[i['label'].isin([500105, 500109, 500113])])
    
for i in control5:
    print(i.head(5))
    


# %%
control5_epochs = []

for i in control5:
    # Filter rows with the desired labels for start and end conditions
    label_start = i[i['label'].isin([6001131, 6001130, 6001050, 6001051, 6001090, 6001091])]
    # label_start = i[i['label'].isin([500105, 500109, 500113])]

    # Initialize lists to hold the timestamps
    start = []
    end = []

    # For each timestamp of the start labels (6001050, 6001051), find the next closest end label (6005050, 6005051)
    for ts_start in label_start['timestamp']:
        # Append the results to the lists
        start.append(ts_start - 2)  # Subtract two seconds before the first number in the sequence
        end.append(ts_start + 10) # if ts_end is not None else end.append(None)  # Add three seconds after the last number in the sequence, if exists

    # Create the DataFrame for the current iteration and append to control5_epochs
    control5_epochs.append(pd.DataFrame({
        'start': start,
        'end': end
    }))

print(control5_epochs) #TODO: print size of each dataframe


# %% [markdown]
# Function to process from a subjects epoch and pupil data to a list of dataframes of the valid trials where missing values have been filled in using linear interpolation:

# %%
def pupil_processing(df_epochs, df_pupil, subject_n):
    valid_trials = []
    excluded_count = 0

    # Iterate over each trial (start and end time)
    for row in df_epochs.itertuples():
        start_time = row.start
        end_time = row.end
        
        # Filter the pupil data for the current trial
        trial_data = df_pupil[(df_pupil['pupil_timestamp'] >= start_time) & (df_pupil['pupil_timestamp'] <= end_time)].copy()
        
        # Calculate Median Absolute Deviation (MAD)
        median_diameter = trial_data['diameter'].median()
        mad_diameter = np.median(np.abs(trial_data['diameter'] - median_diameter))

        # Set values where the absolute deviation is greater than 16 times the MAD to NaN
        mad_threshold = 16
        mad_mask = np.abs(trial_data['diameter'] - median_diameter) > mad_threshold * mad_diameter
        trial_data.loc[mad_mask, 'diameter'] = np.nan

        # Handle blink: Set 'diameter' to NaN where blinks occur
        blink_times = trial_data.loc[trial_data['blink'] == 1, 'pupil_timestamp'].values
        if len(blink_times) > 0:
            lower_bounds = blink_times - 0.1
            upper_bounds = blink_times + 0.1
            blink_mask = np.zeros(len(trial_data), dtype=bool)
            for lower, upper in zip(lower_bounds, upper_bounds):
                blink_mask |= (trial_data['pupil_timestamp'] >= lower) & (trial_data['pupil_timestamp'] <= upper)
            trial_data.loc[blink_mask, 'diameter'] = np.nan
        
        # Remove trials with more than 50% NaN in 'diameter'
        nan_ratio = trial_data['diameter'].isna().mean()
        if nan_ratio <= 0.5:
            # Linear interpolation for missing 'diameter' values
            trial_data.loc[:, 'diameter'] = trial_data['diameter'].interpolate(method='linear')
            
            # Forward fill to handle NaNs at the start
            trial_data.loc[:, 'diameter'] = trial_data['diameter'].ffill().bfill()
        
            # Apply 5-point moving average to the entire data (including interpolated values)
            trial_data.loc[:, 'diameter'] = trial_data['diameter'].rolling(window=5, min_periods=1).mean()

            # Convert 'pupil_timestamp' to a datetime index if not already
            trial_data['pupil_timestamp'] = pd.to_datetime(trial_data['pupil_timestamp'], unit='s')

            # Define the baseline interval (2 seconds from the start)
            start_time = trial_data['pupil_timestamp'].min()
            baseline_interval = trial_data[(trial_data['pupil_timestamp'] >= start_time) & 
                                        (trial_data['pupil_timestamp'] < start_time + pd.Timedelta(seconds=2))]
            
            # Compute the mean of pupil size during the baseline interval
            baseline_mean = baseline_interval['diameter'].mean()

            trial_data.loc[:, 'diameter'] -= baseline_mean

            trial_data.set_index('pupil_timestamp', inplace=True)

            # Resample by using a time-based approach (10 Hz = one data point every 100ms)
            trial_data = trial_data.resample('10ms').mean()

            # Forward fill to handle NaNs at the start
            trial_data.loc[:, 'diameter'] = trial_data['diameter'].ffill().bfill()

            valid_trials.append(trial_data)
        else:
            excluded_count += 1

    print("Trials excluded: {}".format(excluded_count))
    # Step 2: Check if participants have at least 6 valid trials
    if len(valid_trials) < 6:
        print("Participant has fewer than 6 valid trials. Excluding participant.")
        return None
    else:
        print(f"Valid trials count: {len(valid_trials)}")

    # Step 3: Trim each trial to the minimum length
    min_length_subject = min(len(trial) for trial in valid_trials)

    subject_valid_trials = []
    for i, trial in enumerate(valid_trials):
        subject_valid_trials.append(trial.iloc[:min_length_subject])  # Trim to minimum length

    # Initialize lists to store results
    means_subject = []
    sems_subject = []

    # Collect diameter values for each index position across all trials
    for index in range(min_length_subject):
        # Extract diameter values at this index across all trials
        diameter_values_at_index = [trial_data.iloc[index]['diameter'] for trial_data in subject_valid_trials]
        
        # Calculate mean and SEM if there are values (avoid division by zero)
        if len(diameter_values_at_index) > 0:
            mean_diameter = np.mean(diameter_values_at_index)
            sem_diameter = np.std(diameter_values_at_index, ddof=1) / np.sqrt(len(diameter_values_at_index))
        else:
            mean_diameter = np.nan
            sem_diameter = np.nan
        
        # Store results
        means_subject.append(mean_diameter)
        sems_subject.append(sem_diameter)

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Index': range(min_length_subject),
        'Mean_Diameter': means_subject,
        'SEM_Diameter': sems_subject
    })
    print(subject_folders[subject_n])
    # Display the results
    results_df.to_csv(f"Pupil_Processed/pupil_memory_{subject_folders[subject_n]}.csv")
    print("Processing complete.")
    return valid_trials


# %%
all_control5_valid_trials = []

for subject_n in range(len(control5_epochs)):
    print(subject_n)
    processing_result = pupil_processing(control5_epochs[subject_n], all_subjects_pupil[subject_n], subject_n)
    if processing_result:
        all_control5_valid_trials.append(processing_result)


# %%
min_length = min(len(trial) for valid_trials in all_control5_valid_trials for trial in valid_trials)
print(min_length)

final_control5 = []

# Iterate over each valid_trials list and each trial within that list
for valid_trials in all_control5_valid_trials:
    for i, trial in enumerate(valid_trials):
        # Trim each trial to the minimum length by slicing the dataframe
        final_control5.append(trial.iloc[:min_length].reset_index(drop=True))

print(final_control5)


# %%
# Initialize lists to store results
means = []
sems = []

# Collect diameter values for each index position across all trials
for index in range(min_length):
    # Extract diameter values at this index across all trials
    diameter_values_at_index = [trial_data.loc[index, 'diameter'] for trial_data in final_control5]
    
    # Calculate mean and SEM if there are values (avoid division by zero)
    if len(diameter_values_at_index) > 0:
        mean_diameter = np.mean(diameter_values_at_index)
        sem_diameter = np.std(diameter_values_at_index, ddof=1) / np.sqrt(len(diameter_values_at_index))
    else:
        mean_diameter = np.nan
        sem_diameter = np.nan
    
    # Store results
    means.append(mean_diameter)
    sems.append(sem_diameter)

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Index': range(min_length),
    'Mean_Diameter': means,
    'SEM_Diameter': sems
})

# Display the results
print(results_df)
results_df.to_csv("Pupil_Processed/pupil_memory_all.csv")


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

results_df = pd.read_csv("Pupil_Processed/pupil_memory_all.csv")
results_df_control = pd.read_csv("Pupil_Processed/pupil_control_all.csv")

# Assuming results_df is the DataFrame created in the previous step
# Extract the index, mean diameter, and SEM diameter values
indices = results_df['Index']
mean_diameters = (results_df['Mean_Diameter']) / results_df['Mean_Diameter'].std()
sem_diameters = (results_df['SEM_Diameter']) / results_df['Mean_Diameter'].std()

indices_control = results_df_control['Index']
mean_diameters_control = (results_df_control['Mean_Diameter']) / results_df_control['Mean_Diameter'].std()
sem_diameters_control = (results_df_control['SEM_Diameter']) / results_df_control['Mean_Diameter'].std()

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(indices*0.01 - 2, mean_diameters, marker='o', linestyle='-', color='b', label='Memory', markersize=0)
plt.plot(indices*0.01 - 2, mean_diameters_control, marker='o', linestyle='-', color='orange', label='Control', markersize=0)

# Add error shading
plt.fill_between(indices*0.01 - 2, mean_diameters - sem_diameters, mean_diameters + sem_diameters, alpha=0.2, color='blue')
plt.fill_between(indices*0.01 - 2, mean_diameters_control - sem_diameters_control, mean_diameters_control + sem_diameters_control, alpha=0.2, color='orange')

# Add labels and title
plt.xlabel('Time (s)')
plt.ylabel('Baseline Change (standardized)')
plt.title('Mean Pupil Diameter against Time')
plt.axvline(0, color='k', linestyle='--')
plt.legend()
plt.grid(True)

# Set the x-axis ticks
plt.xticks(np.arange(-2, 11, 2))

# Show the plot
plt.show()


# %%



