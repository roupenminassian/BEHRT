from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import pandas as pd

class NTLoader(Dataset):
    def __init__(self, dataframe, max_len):
        self.max_len = max_len
        self.data = dataframe.applymap(self.convert_to_float)
        self.features = ['HR', 'Temp', 'O2Sat', 'Resp']
        self.label_col = 'SepsisLabel'
        self.normalized_data = self.normalize_data(self.data)

    def convert_to_float(self, x):
        try:
            return float(x)
        except ValueError:
            return np.nan  # Convert non-numeric values to NaN

    def normalize_data(self, df):
        df_normalized = df.copy()
        for feature in self.features:
            df_normalized[feature] = df[feature].apply(lambda x: self.normalize_sequence(x) if isinstance(x, list) else x)
        return df_normalized

    def normalize_sequence(self, sequence):
        # Convert to numeric and filter out non-numeric values
        numeric_sequence = pd.to_numeric(sequence, errors='coerce').dropna()

        if len(numeric_sequence) == 0:
            return []

        max_val = numeric_sequence.max()
        min_val = numeric_sequence.min()

        # Normalization
        return [(x - min_val) / (max_val - min_val) if max_val != min_val else 0 for x in numeric_sequence]


    def __getitem__(self, index):
        patient_data = self.normalized_data.iloc[index]
        feature_sequences = [self.pad_sequence(patient_data[feature], 50) for feature in self.features]
        # Ensure the features tensor is of shape (batch_size, sequence_length, num_features)
        features_tensor = torch.tensor(feature_sequences, dtype=torch.float).transpose(0, 1)

        label_sequence = self.pad_sequence(patient_data[self.label_col], pad_value=0)

        # Convert to tensor
        try:
            label_tensor = torch.tensor(label_sequence, dtype=torch.long)
        except RuntimeError as e:
            print("Error with label sequence:", label_sequence)
            raise e

        return features_tensor, label_tensor

    def pad_sequence(self, sequence, pad_value=0):
        # Check if the input is a single value; if so, convert it to a list
        if isinstance(sequence, (float, int, np.float64)):
            sequence = [sequence]

        # Replace 'nan' with pad_value in the sequence
        cleaned_sequence = [x if not np.isnan(x) else pad_value for x in sequence]

        # Pad or truncate the sequence to max_len
        cleaned_sequence += [pad_value] * (self.max_len - len(cleaned_sequence))
        return cleaned_sequence[:self.max_len]

    def __len__(self):
        return len(self.data)