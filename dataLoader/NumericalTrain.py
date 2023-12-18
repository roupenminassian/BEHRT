import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class NTLoader(Dataset):
    def __init__(self, dataframe, max_len):
        self.max_len = max_len
        self.features = ['HR', 'Temp', 'O2Sat', 'Resp']  # Update this list based on actual features
        self.label_col = 'SepsisLabel'
        self.dataframe = dataframe

        # Parse and align sequences
        self.parsed_data = self.parse_and_align_sequences(self.dataframe)

        # Normalize the feature columns
        self.normalized_data = self.normalize_data(self.parsed_data)

    def parse_and_align_sequences(self, df):
        parsed_df = pd.DataFrame(index=df.index)
        for col in df.columns:
            if col in self.features or col == self.label_col:
                parsed_df[col] = df[col].apply(lambda x: self.parse_dict(x))
        aligned_df = pd.DataFrame(index=df.index)
        for col in parsed_df.columns:
            aligned_df[col] = parsed_df[col].apply(lambda x: self.align_sequence(x, self.max_len))
        return aligned_df

    def parse_dict(self, dict_str):
        if pd.isna(dict_str):
            return {}
        else:
            return ast.literal_eval(dict_str)

    def align_sequence(self, sequence, max_len):
        if not sequence:
            return [np.nan] * max_len
        else:
            sorted_sequence = [sequence[k] for k in sorted(sequence)]
            return sorted_sequence + [np.nan] * (max_len - len(sorted_sequence))

    def normalize_data(self, df):
        df_normalized = df.copy()

        # Find the global maximum and minimum values for each feature
        global_max = {feature: np.nanmax(np.concatenate(df[feature].values)) for feature in self.features}
        global_min = {feature: np.nanmin(np.concatenate(df[feature].values)) for feature in self.features}

        # Normalize each feature based on the global max and min
        for feature in self.features:
            max_val = global_max[feature]
            min_val = global_min[feature]
            df_normalized[feature] = df_normalized[feature].apply(
                lambda x: [(v - min_val) / (max_val - min_val) if not np.isnan(v) else 0 for v in x]
            )

        return df_normalized

    def __getitem__(self, index):
        patient_data = self.normalized_data.iloc[index]
        features = [patient_data[feature] for feature in self.features]
        labels = patient_data[self.label_col]

        features_tensor = torch.tensor(features, dtype=torch.float).transpose(0, 1)

        # Replace nan values with 0 in label sequence
        labels = [0 if pd.isna(x) else x for x in labels]

        # Debugging section for labels and normalized features
        #print("Original label sequence:", self.dataframe[self.label_col].iloc[index])
        #print("Processed label sequence before tensor conversion:", labels)
        #print("Normalized features:", features)
        
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return features_tensor, labels_tensor

    def __len__(self):
        return len(self.normalized_data)

# Usage example:
# dataset = NTLoader(your_dataframe, max_len=50)
# dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
