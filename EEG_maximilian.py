# %%
import matplotlib.pyplot as plt
import pandas as pd
import mne
import numpy as np
import os

# Example usage
# subject_numbers = [32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71]  # List of subjects to analyze
subnums = np.arange(32, 99, 1)
subject_numbers = np.delete(subnums, np.where((subnums == 37) | (subnums == 66) | (subnums == 94)))

base_directory = r"C:\Users\Bruger\Documents\DTU\Thesis\Data\ds003838"

# %%

def calculate_theta_power(epochs, channels, baseline):
    """Calculate theta power and SEM for given epochs."""
    freqs = np.arange(4, 9, 1)  # Theta band (4 to 8 Hz), but paper does 1 to 45Hz
    n_cycles = [3.30464978,  3.41292571,  3.52474928, 3.64023672,  3.75950806] # list of five values, one for each frequency (4, 5, 6, 7, 8 Hz, logarithmic scaling)

    # Time-frequency analysis
    power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False)
    power.apply_baseline(baseline=baseline, mode='percent')

    # Average over frequencies, -	Spectral power baseline normalized 
    theta_power_mean = power.data.mean(axis=1)  # Shape: from (n_channels, n_frequencies, n_times) to (n_channels, n_times)
    theta_power_sem = power.data.std(axis=1) / np.sqrt(power.data.shape[1])  # SEM across freqs

    return theta_power_mean, theta_power_sem

def average_frontal_theta_power(subject_numbers, base_directory, tmin=-2, tmax=10, baseline=(-1, 0)):
    all_theta_power_memory = []
    all_theta_power_control = []

    annotated_codes_memory = ['6001050', '6001051', '6001090', '6001091', '6001130', '6001131']
    annotated_codes_control = ['500105', '500109', '500113']
    frontal_midline_channels = ['AFz', 'AF3', 'AF4', 'Fz', 'F1', 'F2', 'F3', 'F4', 'FC3', 'FC1', 'FC2', 'FC4', 'Cz', 'C3', 'C1', 'C2', 'C4']

    for subject in subject_numbers:
        file_path = os.path.join(base_directory, f"sub-{subject:03d}", "eeg", f"sub-{subject:03d}_task-memory_eeg.set")
        
        
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
        raw.set_eeg_reference(ref_channels="average") # Re-reference data to average reference
        raw.filter(l_freq=4.0, h_freq=8.0) #Band pass filter, why different?
        
        # Get events
        events, event_ids = mne.events_from_annotations(raw)

        # Create a mapping for events
        event_mapping_memory = {code: event_ids[code] for code in annotated_codes_memory if code in event_ids}
        event_mapping_control = {code: event_ids[code] for code in annotated_codes_control if code in event_ids}

        epochs_memory = mne.Epochs(raw, events, event_id=event_mapping_memory, tmin=tmin, tmax=tmax, preload=True) # epochs data, but from -2 to 10 seconds insetad of [-1.5 to 3.5]
        epochs_control = mne.Epochs(raw, events, event_id=event_mapping_control, tmin=tmin, tmax=tmax, preload=True)

        # mne.preprocessing.compute_current_source_density(
        #     epochs_memory,                # Input data (MNE epochs object)
        #     sphere='auto',                # Auto-sphere estimation, or you can provide a custom sphere model
        #     lambda2=1e-5,                 # Smoothing constant (λ = 10–5)
        #     stiffness=4,                  # Stiffness parameter (m = 4)
        #     n_legendre_terms=50,          # Number of iterations (Legendre terms = 50)
        #     copy=False,                    # Whether to apply in-place or return a modified copy
        #     verbose=True                  # Optional: to view additional details during execution
        # )

        # mne.preprocessing.compute_current_source_density(
        #     epochs_control,                # Input data (MNE epochs object)
        #     sphere='auto',                # Auto-sphere estimation, or you can provide a custom sphere model
        #     lambda2=1e-5,                 # Smoothing constant (λ = 10–5)
        #     stiffness=4,                  # Stiffness parameter (m = 4)
        #     n_legendre_terms=50,          # Number of iterations (Legendre terms = 50)
        #     copy=False,                    # Whether to apply in-place or return a modified copy
        #     verbose=True                  # Optional: to view additional details during execution
        # )

        print(f"Subject {subject:03d}: {len(epochs_memory)} memory epochs, {len(epochs_control)} control epochs")

        # Calculate theta power and SEM
        theta_power_memory, sem_memory = calculate_theta_power(epochs_memory.copy().pick_channels(frontal_midline_channels), frontal_midline_channels, baseline)
        theta_power_control, sem_control = calculate_theta_power(epochs_control.copy().pick_channels(frontal_midline_channels), frontal_midline_channels, baseline)

        all_theta_power_memory.append(theta_power_memory)
        all_theta_power_control.append(theta_power_control)

        # Create DataFrame for subject-level results
        times = np.linspace(tmin, tmax, theta_power_memory.shape[1])  # Shape: (n_times,)
        subject_theta_df = pd.DataFrame({
            'Time': times,
        })

        for channel_idx, channel in enumerate(frontal_midline_channels):
            subject_theta_df[f'Theta_Power_Memory_{channel}'] = theta_power_memory[channel_idx]
            subject_theta_df[f'SEM_Memory_{channel}'] = sem_memory[channel_idx]
            subject_theta_df[f'Theta_Power_Control_{channel}'] = theta_power_control[channel_idx]
            subject_theta_df[f'SEM_Control_{channel}'] = sem_control[channel_idx]

        subject_theta_df.to_csv(f"Theta_Processed/theta_results_subject_{subject:03d}.csv", index=False)

    # Average theta power across subjects
    if all_theta_power_memory and all_theta_power_control:
        theta_power_avg_memory = np.mean(all_theta_power_memory, axis=0)
        theta_power_avg_control = np.mean(all_theta_power_control, axis=0)
        theta_power_sem_memory = np.std(all_theta_power_memory, axis=0) / np.sqrt(len(subject_numbers))
        theta_power_sem_control = np.std(all_theta_power_control, axis=0) / np.sqrt(len(subject_numbers))

        # Save averaged results
        avg_theta_results_df = pd.DataFrame({
            'Time': times,
        })

        for channel_idx, channel in enumerate(frontal_midline_channels):
            avg_theta_results_df[f'Mean_Power_Memory_{channel}'] = theta_power_avg_memory[channel_idx]
            avg_theta_results_df[f'SEM_Power_Memory_{channel}'] = theta_power_sem_memory[channel_idx]
            avg_theta_results_df[f'Mean_Power_Control_{channel}'] = theta_power_avg_control[channel_idx]
            avg_theta_results_df[f'SEM_Power_Control_{channel}'] = theta_power_sem_control[channel_idx]

        avg_theta_results_df.to_csv("Theta_Processed/theta_results_all.csv", index=False)

        # Plot averaged theta power across subjects
        plt.figure(figsize=(10, 6))
        for channel_idx, channel in enumerate(frontal_midline_channels):
            plt.plot(times, theta_power_avg_memory[channel_idx], label=f'Averaged Theta Power (Memory - {channel})')
            plt.fill_between(times, theta_power_avg_memory[channel_idx] - theta_power_sem_memory[channel_idx],
                             theta_power_avg_memory[channel_idx] + theta_power_sem_memory[channel_idx], alpha=0.3)

            plt.plot(times, theta_power_avg_control[channel_idx], label=f'Averaged Theta Power (Control - {channel})')
            plt.fill_between(times, theta_power_avg_control[channel_idx] - theta_power_sem_control[channel_idx],
                             theta_power_avg_control[channel_idx] + theta_power_sem_control[channel_idx], alpha=0.3)

        plt.xlabel('Time (s)')
        plt.ylabel('Theta Power (Percent change)')
        plt.title('Average Frontal Midline Theta Power (4-8 Hz)')
        plt.xticks(np.arange(tmin, tmax + 1, 2))
        plt.axvline(0, color='k', linestyle='--')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
    else:
        print("No theta power data available to average.")

average_frontal_theta_power(subject_numbers, base_directory)


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load theta power data
memory_data = pd.read_csv("Theta_Processed/theta_results_all.csv", usecols=['Mean_Power_Memory_AFz', 'SEM_Power_Memory_AFz'])
control_data = pd.read_csv("Theta_Processed/theta_results_all.csv", usecols=['Mean_Power_Control_AFz', 'SEM_Power_Control_AFz'])

# Resample: downsample the data by averaging every 10 rows
memory_resampled = memory_data.groupby(np.arange(len(memory_data)) // 10).mean().reset_index(drop=True)
control_resampled = control_data.groupby(np.arange(len(control_data)) // 10).mean().reset_index(drop=True)

# Normalize the theta power columns (z-scoring)
memory_resampled['Power_Normalized'] = (memory_resampled['Mean_Power_Memory_AFz'] - memory_resampled['Mean_Power_Memory_AFz'].mean()) / memory_resampled['Mean_Power_Memory_AFz'].std()
memory_resampled['SEM_Normalized'] = memory_resampled['SEM_Power_Memory_AFz'] / memory_resampled['Mean_Power_Memory_AFz'].std()

control_resampled['Power_Normalized'] = (control_resampled['Mean_Power_Control_AFz'] - control_resampled['Mean_Power_Control_AFz'].mean()) / control_resampled['Mean_Power_Control_AFz'].std()
control_resampled['SEM_Normalized'] = control_resampled['SEM_Power_Control_AFz'] / control_resampled['Mean_Power_Control_AFz'].std()

# Define time indices for plotting
time_axis = np.arange(len(memory_resampled)) * 0.01 - 2  # Adjust based on 10 ms intervals and offset for plotting

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(time_axis, memory_resampled['Power_Normalized'], marker='o', linestyle='-', color='b', label='Memory', markersize=0)
plt.plot(time_axis, control_resampled['Power_Normalized'], marker='o', linestyle='-', color='orange', label='Control', markersize=0)

# Add error shading
plt.fill_between(time_axis, 
                 memory_resampled['Power_Normalized'] - memory_resampled['SEM_Normalized'], 
                 memory_resampled['Power_Normalized'] + memory_resampled['SEM_Normalized'], alpha=0.2, color='blue')
plt.fill_between(time_axis, 
                 control_resampled['Power_Normalized'] - control_resampled['SEM_Normalized'], 
                 control_resampled['Power_Normalized'] + control_resampled['SEM_Normalized'], alpha=0.2, color='orange')

# Add labels and title
plt.xlabel('Time (s)')
plt.ylabel('Baseline Change (standardized)')
plt.title('AFz - Theta Power against Time')
plt.axvline(0, color='k', linestyle='--')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


# %%



