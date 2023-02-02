
import time
import sdl2
import math
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, AggOperations, FilterTypes, WindowOperations, NoiseTypes

from scipy import signal

import dearpygui.dearpygui as dpg

board_id = BoardIds.CROWN_BOARD.value
# board_id = BoardIds.SYNTHETIC_BOARD.value
params = BrainFlowInputParams()
params.mac_address = "C0:EE:40:84:DD:56"
params.serial_number = "58a99b0107e64cd40ea5e6607882cbe2"
params.board_id = board_id
params.timeout = 5
BoardShim.enable_dev_board_logger()
board = BoardShim(board_id, params)
board.prepare_session()

sdl2.SDL_Init(sdl2.SDL_INIT_JOYSTICK)
joystick = sdl2.SDL_JoystickOpen(0)

def update_series(data):
    for i, channel in enumerate(data):
        dpg.set_value(f'c{i}', [np.arange(0, len(channel)), channel])
        dpg.fit_axis_data(f'y_axis_c{i}')

signal_length = 256 * 5 // 2

dpg.create_context()
dpg.create_viewport(title="Recording Data", width=600, height=800)
dpg.setup_dearpygui()

with dpg.window(label="Recording Data", tag="Primary Window"):
    for i in range(8):
        with dpg.plot(height=75, width=-1, no_menus=True, no_mouse_pos=True):
            dpg.add_plot_axis(dpg.mvXAxis, tag=f"x_axis_c{i}", no_gridlines=True, no_tick_marks=True, no_tick_labels=True)
            dpg.add_plot_axis(dpg.mvYAxis, tag=f"y_axis_c{i}", no_gridlines=True, no_tick_marks=True, no_tick_labels=True)
            
            dpg.add_line_series(np.arange(0, signal_length), np.zeros((signal_length)), label="mV", parent=f"y_axis_c{i}", tag=f"c{i}")
            dpg.set_axis_limits(f'x_axis_c{i}', 0, signal_length // 2)
            dpg.fit_axis_data(f'y_axis_c{i}')
    dpg.add_text("X: None Y: None Direction: None", tag="direction_text")

dpg.set_primary_window("Primary Window", True)

raw_data = np.empty((8, 0), dtype=np.float64)

start_time = time.time()
BoardShim.disable_board_logger()
board.start_stream()
time.sleep(6)
dpg.show_viewport()
while(True):
    data = board.get_board_data()[1:9]
    
    if data.shape[1] != 0:
        raw_data = np.concatenate((raw_data, data), axis=1)
        
        gui_data = raw_data.copy()
        gui_data = mne.filter.resample(gui_data, down=2)
        gui_data = mne.filter.notch_filter(gui_data, 128, 60, method='iir', verbose=False)
        gui_data = mne.filter.filter_data(gui_data, 128, 1.0, 50.0, method='iir', verbose=False)
            # DataFilter.perform_bandpass(gui_data[i], 256//2, 1.0, 50.0, 4, FilterTypes.BUTTERWORTH.value, 0)
            # DataFilter.remove_environmental_noise(gui_data[i], 256//2, NoiseTypes.SIXTY.value)
            # for hz in range(60, 121, 60):
            #     b, a = signal.iirnotch(hz, 10, 128)
            #     gui_data[i] = signal.filtfilt(b, a, gui_data[i]) # Americas wires run at 60Hz; Because wires run at 60Hz, there is a 120Hz harmonic
        
        update_series(gui_data[:, -signal_length:])
    dpg.render_dearpygui_frame()
    time.sleep(0.05)

board.release_session()


