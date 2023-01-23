import os
import numpy as np

from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, NoiseTypes, WindowOperations

from scipy import fft
from scipy import signal

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot


class BCIDataset(Dataset):
    def __init__(self, folder, hz=256, seconds=1.75, one_hot=True):
        self.hz = hz
        self.seconds = seconds
        
        self.one_hot = one_hot
        
        files = os.listdir(folder)
        self.dataset = np.load(os.path.join(folder, files[0]), allow_pickle=True)
        files.pop(0)
        for file_path in files:
            self.dataset = np.concatenate((self.dataset, np.load(os.path.join(folder, file_path), allow_pickle=True)), axis=1)
            
        self.inputs, self.targets = self.format_dataset(self.dataset)

    def filter_data(self, data):
        DataFilter.perform_bandpass(data, self.hz, 1.0, 50.0, 4, FilterTypes.BUTTERWORTH.value, 0)
        # DataFilter.remove_environmental_noise(data, self.hz, NoiseTypes.SIXTY.value)
        for i in range(60, 121, 60):
            b, a = signal.iirnotch(i, 10, self.hz)
            data[:] = signal.filtfilt(b, a, data) # Americas wires run at 60Hz; Because wires run at 60Hz, there is a 120Hz harmonic

    def format_dataset(self, dataset):
        hz = self.hz
        length = int((hz // 2) * self.seconds)
        
        inputs = []
        targets = []
        for i in range(len(dataset[0])):
            session = np.empty((8, len(dataset[0][i][0]) // 2))
            for j in range(len(dataset[0][i])):
                self.filter_data(dataset[0][i][j])
                session[j] = DataFilter.perform_downsampling(dataset[0][i][j], 2, AggOperations.MEAN.value)
            
            for j in range(0, len(session[0]) - int(hz / 2 * self.seconds), int((hz // 2) * (self.seconds / 4))): # Gets 1.75 seconds of data and uses a 75% overlap
                data = np.array(session[:, j:j + length])
                
                inp = np.empty((8, length // 2))
                for k, channel in enumerate(data):
                    inp[k] = fft.rfft(channel)[:-1]
            
                inputs.append(inp)
                targets.append(dataset[1][i])

        targets = np.array(targets)
        inputs = np.array(inputs)
        
        return inputs, targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.one_hot:
            sample = {'inputs': torch.tensor(self.inputs[idx], dtype=torch.float32), 'targets': one_hot(torch.tensor(self.targets[idx], dtype=torch.int64), num_classes=3).type(torch.float32)}
        else:
            sample = {'inputs': torch.tensor(self.inputs[idx], dtype=torch.float32), 'targets': torch.tensor(self.targets[idx], dtype=torch.int64)}

        return sample