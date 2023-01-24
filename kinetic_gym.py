import sdl2
import time
import numpy as np
import torch

import gym
from gym import spaces

from scipy import signal
from scipy import fft

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, AggOperations, FilterTypes, WindowOperations, NoiseTypes


class KineticEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(KineticEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-8000, high=8000,
                                            shape=(8, 64), dtype=np.float32)

        sdl2.SDL_Init(sdl2.SDL_INIT_JOYSTICK)
        self.joystick = sdl2.SDL_JoystickOpen(0)
        
        self.setup_eeg()
        self.hz = 256
        time.sleep(3)
        
        self.raw_data = np.zeros((8, 256), dtype=np.float64)
        
        self.real_direction = 0
        self.predicted_direction = 0
        self.past_predictions = []

    def setup_eeg(self):
        board_id = BoardIds.CROWN_BOARD.value
        params = BrainFlowInputParams()
        params.mac_address = "C0:EE:40:84:DD:56"
        params.serial_number = "58a99b0107e64cd40ea5e6607882cbe2"
        params.board_id = board_id
        params.timeout = 5
        BoardShim.enable_dev_board_logger()
        self.board = BoardShim(board_id, params)
        self.board.prepare_session()
        BoardShim.disable_board_logger()
        self.board.start_stream()

    def update_eeg(self):
        new_data = self.board.get_board_data()[1:9]
        if(new_data.shape[1] <= 0):
            return
        np.roll(self.raw_data, -new_data.shape[1], axis=1)
        self.raw_data[:, -new_data.shape[1]:] = new_data[:, -256:]
        
    def filter_data(self):
        data = np.copy(self.raw_data)
        # b, a = signal.butter(4, [1.0, 128.0], btype='bandpass', analog=True)
        # data = signal.filtfilt(b, a, data)
        for i in range(len(data)):
            DataFilter.perform_bandpass(data[i], self.hz, 1.0, 50.0, 4, FilterTypes.BUTTERWORTH.value, 0)
        
        for i in range(60, 121, 60):
            b, a = signal.iirnotch(i, 10, self.hz)
            data = signal.filtfilt(b, a, data) # Americas wires run at 60Hz; Because wires run at 60Hz, there is a 120Hz harmonic
        
        data = np.average(data.reshape(8, -1, 2), axis=2)
        
        data = fft.rfft(data, axis=1)[:, :-1]
        return data

    def get_direction(self):
        sdl2.SDL_PumpEvents()
        # depending on the gamepad this gives you a value between -32768 and +32768
        # or between 0 and 32768
        x = sdl2.SDL_JoystickGetAxis(self.joystick, 0)
        y = sdl2.SDL_JoystickGetAxis(self.joystick, 1)
        # x_direction = "right" if joy_x / 32768 > 0.3 else "left" if joy_x / 32768 < -0.1 else "center" # WARNING: A bit offset because of drift, should be 0.2 and -0.2 for the deadzone
        # print(f"X: {joy_x} Y: {joy_y} Direction: {x_direction}                    ", end="\r")

        return x, y

    def step(self, action):
        x, y = self.get_direction()
        
        time.sleep(0.1)
        
        # WARNING: A bit offset because of drift, should be 0.2 and -0.2 for the deadzone
        self.real_direction = 2 if x / 32768 > 0.3 else 0 if x / 32768 < -0.1 else 1
        self.predicted_direction = action
        self.past_predictions.append([self.real_direction, action])
        
        if(action == self.real_direction):
            reward = 1
        else:
            reward = -1
        
        self.update_eeg()
        observation = torch.from_numpy(self.filter_data()).type(torch.float32)
        
        done = False
        info = {}
        
        return observation, reward, done, info

    def reset(self):
        self.update_eeg()
        observation = self.filter_data()
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        real_direction = "right" if self.real_direction == 2 else "left" if self.real_direction == 0 else "center"
        predicted_direction = "right" if self.predicted_direction == 2 else "left" if self.predicted_direction == 0 else "center"
        print(f"Current Direction: {real_direction}")
        print(f"Predicted Direction: {predicted_direction}")
        print(f"Accuracy: {np.mean(np.array(self.past_predictions)[-25:, 0] == np.array(self.past_predictions)[-25:, 1])}")

    def close(self):
        self.board.stop_stream()
