import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import FastICA



def load_dataset(USED_DATASET, TRAIN_AMOUNT, NEED_NOISE, NEED_VALID):
    VAL_AMOUNT = 1.0 - TRAIN_AMOUNT

    path = 'datasets/swat/'  # 'datasets/swat/'
    if USED_DATASET == 'swat':

        # Path configuration
        base_path = path
        normal_data_file = 'SWaT_Dataset_Normal_v1.csv'
        attack_data_file = 'SWaT_Dataset_Attack_v0.csv'

        # Load SWaT normal dataset
        print("Loading SWaT dataset")
        tqdm.pandas(desc="Loading TRAIN", unit='sample')
        train = pd.read_csv(os.path.join(base_path, normal_data_file), low_memory=False)

        # Strip whitespace from column names
        train.columns = [col.strip() for col in train.columns]

        # Process timestamps and handle missing values
        train['Timestamp'] = train['Timestamp'].progress_apply(lambda x: 1 if x == 'Yes' else 0)
        train.dropna(inplace=True)
        train.drop(columns='Timestamp', inplace=True)

        # Handling noise in the dataset
        if not NEED_NOISE:
            train = train.iloc[21600:, :]

        # Load SWaT attack dataset
        tqdm.pandas(desc="Loading TEST", unit='sample')
        test = pd.read_csv(os.path.join(base_path, attack_data_file), low_memory=False)

        # Strip whitespace from column names
        test.columns = [col.strip() for col in test.columns]

        # Process timestamps and handle missing values
        test['Timestamp'] = test['Timestamp'].progress_apply(lambda x: 1 if x == 'Yes' else 0)
        test.dropna(inplace=True)
        test.drop(columns='Timestamp', inplace=True)

        # Replace labels from 'Normal'/'Attack' to 1/-1
        print('Replacing Labels')
        label_mapping = {
            'Normal': 1,
            'Attack': -1,
            'A ttack': -1  # Handling typo
        }
        for dataframe in [train, test]:
            dataframe['Normal/Attack'] = dataframe['Normal/Attack'].replace(label_mapping).astype(int)

        # Optionally split the train dataset for validation
        if NEED_VALID and VAL_AMOUNT is not None:
            val_start_index = int(len(train) * (1 - VAL_AMOUNT))
            val = train.iloc[val_start_index:]
            train = train.iloc[:val_start_index]

        # Compute dataset statistics
        mean = train.mean()[:-1]
        std = train.std()[:-1]
        input_dim = len(train.columns) - 1
        print('Finished')
        if NEED_VALID:
            return train, val, test, mean, std, input_dim
        else:
            return train, test, mean, std, input_dim

    #--------------------------------WADI---------------------------------------------------------
    elif USED_DATASET == 'wadi':

        print("Loading WADI dataset")  # WADI DATASET from iTrust
        path = 'datasets/WADI/'

        # Loading WADI train data
        tqdm.pandas(desc="Loading TRAIN", unit=' sample')
        train = pd.read_csv(os.path.join(path, 'WADI_14days_new.csv'),
                            skiprows=[1, 2, 3],
                            skip_blank_lines=True,
                            low_memory=False)
        for col in train.columns:
            if train[col].dtype == 'float64':
                train[col] = train[col].astype('float32')
            elif train[col].dtype == 'int64':
                train[col] = train[col].astype('int32')

        # Strip whitespace from column names
        train.columns = [col.strip() for col in train.columns]

        train = train.dropna()
        train['Normal/Attack'] = 1

        # Loading WADI test data
        tqdm.pandas(desc="Loading TEST", unit=' sample')
        test = pd.read_csv(os.path.join(path, 'WADI_attackdataLABLE.csv'),
                           low_memory=False)
        for col in test.columns:
            if test[col].dtype == 'float64':
                test[col] = test[col].astype('float32')
            elif test[col].dtype == 'int64':
                test[col] = test[col].astype('int32')

        # Strip whitespace from column names
        test.columns = [col.strip() for col in test.columns]

        test = test.dropna()
        test.rename(columns={'Attack LABLE (1:No Attack, -1:Attack)': 'Normal/Attack'}, inplace=True)

        # Setting index columns for train and test
        print("Setting index for TRAIN")
        train.drop(columns=['Date', 'Time'], inplace=True)
        train = train.set_index('Row')

        print("Setting index for TEST")
        test.drop(columns=['Date', 'Time'], inplace=True)
        test = test.set_index('Row')

        if NEED_VALID:
            train = train.iloc[:int(len(train) * VAL_AMOUNT)]
            val = train.iloc[int(len(train) * VAL_AMOUNT):]

        # Add noise to the training dataset if needed
        if NEED_NOISE:
            std = train.std()
            num_points_to_modify = int(len(train) * 0.05)
            features_to_modify = train.columns[:-1]  # Exclude 'Normal/Attack'
            noise_scale = 0.5 * std[features_to_modify]

            for feature in features_to_modify:
                if std[feature] != 0:
                    noise = np.random.normal(0, noise_scale[feature], num_points_to_modify)
                    train.loc[train.index[:num_points_to_modify], feature] = np.maximum(
                        0, train.loc[train.index[:num_points_to_modify], feature] + noise)

        mean = train.mean()
        std = train.std()
        input_dim = len(train.columns) - 1

        if NEED_VALID:
            return train, val, test, np.array(mean), np.array(std), input_dim
        else:
            return train, test, np.array(mean), np.array(std), input_dim



class MyDataset(Dataset):
    def __init__(self, df, seq_length, shift_length, FS='none', is_train=True,
                 normalization='StandardScaler', median=None, mad_values=None, scaler=None):

        self.FS = FS
        self.df = df.copy()  # Create a copy to avoid modifying the original DataFrame
        self.seq_length = seq_length
        self.shift_length = shift_length
        self.is_train = is_train
        self.normalization = normalization
        self.median = median  # Set default values to avoid AttributeError
        self.mad_values = mad_values
        self.scaler = scaler

        # Rename and clean label column for consistency
        self.df.columns = [col.strip() for col in self.df.columns]
        self.df.rename(columns={'Attack LABLE (1:No Attack, -1:Attack)': 'Normal/Attack'}, inplace=True)

        # Define features and label
        label_column = 'Normal/Attack'
        self.label_column = label_column
        features = [col for col in self.df.columns if col != label_column]

        if self.normalization == 'MinMax':
            if is_train:
                # Fit and transform training data with StandardScaler
                self.scaler = MinMaxScaler()
                self.df[features] = self.scaler.fit_transform(self.df[features])
            else:
                # Use provided scaler for test/validation data
                if self.scaler is None:
                    raise ValueError("For test data, scaler must be provided if normalization is 'StandardScaler'.")
                self.df[features] = self.scaler.fit_transform(self.df[features])

        # Apply Feature Selection (FastICA) if specified
        if self.FS == 'fastica':
            nb_feature = len(features)
            self.ica = FastICA(n_components=nb_feature, random_state=42, max_iter=1000)
            transformed_data = self.ica.fit_transform(self.df[features]) if is_train else self.ica.transform(self.df[features])
            self.data = transformed_data.tolist()
            print(f"FastICA applied. New data shape: {transformed_data.shape}")
        else:
            self.data = self.df[features].values.tolist()

        # Prepare Labels
        self.label = self.df[label_column].values.tolist()

        # Create Sequence Indices
        self.indices = np.arange(0, len(self.data) - self.seq_length + 1, self.shift_length)
        print(f"{'Training' if is_train else 'Test'} Dataset initialized. Number of sequences: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        data_seq = self.data[index:index + self.seq_length]
        label_seq = self.label[index:index + self.seq_length]

        data_tensor = torch.FloatTensor(data_seq)
        label_tensor = torch.LongTensor(label_seq)

        return data_tensor, label_tensor



def generate_labels(test_loader, device, anomaly_criteria='one'):
    """
    Generate labels for each sliding window in the test_loader based on the anomaly criteria.

    Args:
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to perform computations on.
        anomaly_criteria (str): Criteria for labeling windows as anomalous ('one' or 'half').

    Returns:
        list: List of labels for each sliding window.
        list: List of processed test data with concatenated target information.
    """
    target_list = []
    testing_arr = []

    for inputs, targets in tqdm(test_loader, desc="Processing test data"):
        # Perform computations on CPU to save GPU memory
        inputs = inputs.cpu()
        targets = targets.cpu()

        # Determine the labeling criterion
        if anomaly_criteria == 'one':
            # Label as anomalous (-1) if any element in the window is anomalous (-1)
            l = torch.where(targets.eq(-1).any(dim=1), torch.tensor(-1), torch.tensor(1))
        elif anomaly_criteria == 'half':
            # Label as anomalous (-1) if more than half of the elements in the window are anomalous (-1)
            num_anomalies = targets.eq(-1).sum(dim=1)
            half_window_size = math.ceil(targets.size(1) / 2)
            l = torch.where(num_anomalies >= half_window_size, torch.tensor(-1), torch.tensor(1))
        else:
            raise ValueError("anomaly_criteria must be 'one' or 'half'")

        # Concatenate targets to each test data element
        inputs_with_targets = torch.cat((inputs, targets.unsqueeze(2)), dim=2)

        # Process data immediately
        target_list.extend(l.tolist())
        # Convert inputs_with_targets to NumPy arrays and extend the list
        testing_arr.extend(inputs_with_targets.numpy())

    return target_list, testing_arr
