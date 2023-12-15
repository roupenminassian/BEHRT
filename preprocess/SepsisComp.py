import pandas as pd
import os

def read_aggregate_and_save(directory, save_directory, columns_of_interest):
    all_patients_data = []

    for filename in os.listdir(directory):
        if filename.endswith('.psv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path, sep='|', usecols=columns_of_interest)

            # Remove rows where any of the columns of interest is NaN
            df = df.dropna(subset=columns_of_interest)

            if df.empty:
                print(f"No valid data in file: {filename}. Skipping this file.")
                continue

            # Aggregate each column into a list
            aggregated_data = df[columns_of_interest].agg(list).to_dict()
            aggregated_data['PatientID'] = filename.replace('.psv', '')  # Extract patient ID from filename

            all_patients_data.append(aggregated_data)

    # Combine all patient data into a single DataFrame
    combined_data = pd.DataFrame(all_patients_data)

    # Save the combined DataFrame
    combined_save_path = os.path.join(save_directory, 'final_dataset.csv')
    combined_data.to_csv(combined_save_path, index=False)

# Columns of interest
columns_of_interest = ['HR', 'Temp', 'O2Sat', 'Resp', 'SepsisLabel']

# Usage
directory = '../all_dataset'  # Replace with the path to your .psv files
save_directory = 'final_dataset'  # Replace with your desired save directory
read_aggregate_and_save(directory, save_directory, columns_of_interest)