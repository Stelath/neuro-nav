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
    def __init__(self, folder, hz=256, seconds=1, overlap=0.5, one_hot=True):
        self.hz = hz
        self.seconds = seconds
        self.overlap = overlap
        
        self.one_hot = one_hot
        
        files = os.listdir(folder)
        self.dataset = np.load(os.path.join(folder, files[0]), allow_pickle=True)
        files.pop(0)
        for file_path in files:
            ds = np.load(os.path.join(folder, file_path), allow_pickle=True)
            self.dataset[0] = np.concatenate((self.dataset[0], ds[0]), axis=1)
            self.dataset[1] = np.concatenate((self.dataset[1], ds[1]), axis=0)
            
        self.inputs, self.targets = self.format_dataset(self.dataset)
        self.rebalance()

    def filter_data(self, data):
        DataFilter.perform_bandpass(data, self.hz, 1.0, 100.0, 4, FilterTypes.BUTTERWORTH.value, 0)
        # DataFilter.remove_environmental_noise(data, self.hz, NoiseTypes.SIXTY.value)
        for i in range(60, 121, 60):
            b, a = signal.iirnotch(i, 10, self.hz)
            data[:] = signal.filtfilt(b, a, data) # Americas wires run at 60Hz; Because wires run at 60Hz, there is a 120Hz harmonic

    def format_dataset(self, dataset):
        hz = self.hz
        length = int((hz // 2) * self.seconds)
        
        inputs = []
        targets = []
        session = np.empty((8, len(dataset[0][0]) // 2))
        for i in range(len(dataset[0])):
            self.filter_data(dataset[0][i])
            session[i] = DataFilter.perform_downsampling(dataset[0][i], 2, AggOperations.MEAN.value)
        
        overlap = 1 - self.overlap
        skip = int((hz // 2) * (self.seconds * overlap))
        for i in range(0, len(session[0]) - int(hz / 2 * self.seconds), skip): # Gets 1 seconds of data and uses a 75% overlap
            data = np.array(session[:, i:i + length])
            
            # inp = np.empty((8, length // 2))
            # for k, channel in enumerate(data):
            #     inp[k] = fft.rfft(channel)[:-1]
            
            inp = np.empty((8, self.hz//2, length))
            for k, channel in enumerate(data):
                inp[k] = signal.cwt(channel, signal.ricker, np.arange(1, self.hz//2 + 1))
        
            inputs.append(inp)
            # inputs.append(data)
            targets.append(dataset[1][2 * (i + skip)])

        targets = np.array(targets)
        inputs = np.array(inputs)
        
        return inputs, targets

    def rebalance(self):
        left = 0
        center = 0
        right = 0
        for target in self.targets:
            if target == 0:
                left += 1
            elif target == 1:
                center += 1
            elif target == 2:
                right += 1

        lowest = min(left, center, right)
        # Print left center right and lowest
        
        for i in range(len(self.inputs) - (left + center + right) + 3 * lowest):
            if left > lowest and self.targets[i] == 0:
                self.inputs = np.delete(self.inputs, i, axis=0)
                self.targets = np.delete(self.targets, i, axis=0)
                i -= 1
                left -= 1
            elif center > lowest and self.targets[i] == 1:
                self.inputs = np.delete(self.inputs, i, axis=0)
                self.targets = np.delete(self.targets, i, axis=0)
                i -= 1
                center -= 1
            elif right > lowest and self.targets[i] == 2:
                self.inputs = np.delete(self.inputs, i, axis=0)
                self.targets = np.delete(self.targets, i, axis=0)
                i -= 1
                right -= 1

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.one_hot:
            sample = {'inputs': torch.tensor(self.inputs[idx], dtype=torch.float32), 'targets': one_hot(torch.tensor(self.targets[idx], dtype=torch.int64), num_classes=3).type(torch.float32)}
        else:
            sample = {'inputs': torch.tensor(self.inputs[idx], dtype=torch.float32), 'targets': torch.tensor(self.targets[idx], dtype=torch.int64)}

        return sample