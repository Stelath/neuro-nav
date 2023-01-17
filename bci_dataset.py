import os
import numpy as np

from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, NoiseTypes, WindowOperations

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot


class BCIDataset(Dataset):
    def __init__(self, np_file, data_length=256):
        self.data_length = data_length
        
        self.dataset = np.load(np_file, allow_pickle=True)
        self.inputs, self.targets = self.format_dataset(self.dataset)

    def filter_data(self, data):
        sr = BoardShim.get_sampling_rate(BoardIds.CROWN_BOARD.value)

        DataFilter.perform_bandpass(data, sr, 5.0, 50.0, 4,FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.remove_environmental_noise(data, sr, NoiseTypes.SIXTY.value) # Americas wires run at 60Hz

    def format_dataset(self, dataset):
        hz = 256
        inputs = []
        targets = []
        for i in range(len(dataset[0])):
            session = np.empty((8, len(dataset[0][i][0]) // 2))
            for j in range(len(dataset[0][i])):
                self.filter_data(dataset[0][i][j])
                session[j] = DataFilter.perform_downsampling(dataset[0][i][j], 2, AggOperations.MEAN.value)
            
            for j in range(len(session[0]) - hz):
                data = np.array(session[:, j:j + hz])
                
                inp = np.empty((8, hz // 4))
                for k, channel in enumerate(data):
                    inp[k] = DataFilter.perform_fft(channel[: hz // 2], WindowOperations.NO_WINDOW.value)[:hz // 4]
            
                inputs.append(inp)
                targets.append(dataset[1][i])

        targets = np.array(targets)
        inputs = np.array(inputs)
        
        return inputs, targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {'inputs': torch.tensor(self.inputs[idx], dtype=torch.float32), 'targets': one_hot(torch.tensor(self.targets[idx], dtype=torch.int64), num_classes=3).type(torch.float32)}

        return sample