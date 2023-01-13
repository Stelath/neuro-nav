import os
import numpy as np

from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot


class BCIDataset(Dataset):
    def __init__(self, np_file, data_length=256):
        self.data_length = data_length
        
        self.dataset = np.load(np_file, allow_pickle=True)
        self.inputs, self.targets = self.format_dataset(self.dataset)

    def filter_data(data):
        sr = BoardShim.get_sampling_rate(BoardIds.CROWN_BOARD.value)

        DataFilter.perform_bandpass(data, sr, 5.0, 50.0, 4,FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.remove_environmental_noise(data, sr, NoiseTypes.SIXTY.value) # Americas wires run at 60Hz

    def format_dataset(self, dataset):
        inputs = []
        targets = []
        for i in range(len(dataset[0])):
            for j in range(len(dataset[0][i])):
                self.filter_data(dataset[0][i][j])
            
            data = np.array(dataset[0][i])
            data = np.mean(data, axis=0)
            
            length = data.shape[1]
            excess = length % self.data_length
            
            # Cut off excess data
            if excess != 0:
                data = data[:,excess:]
            length = data.shape[1]
            data = np.split(data, length // self.data_length, axis=1)
            
            # Add to inputs and outputs
            for split in data:
                inputs.append(split)
                targets.append(dataset[1][i])

        targets = np.array(targets)
        inputs = np.array(inputs)
        
        return inputs, targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {'inputs': torch.tensor(self.inputs[idx], dtype=torch.float32), 'targets': one_hot(torch.tensor(self.targets[idx], dtype=torch.int64), num_classes=3)}

        return sample