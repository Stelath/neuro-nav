import time

import brainflow
from brainflow import BoardIds
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

def main ():
    board_id = BoardIds.CROWN_BOARD.value # or BoardIds.NOTION_2_BOARD.value or BoardIds.NOTION_1_BOARD.value
    params = BrainFlowInputParams()
    params.mac_address = "C0:EE:40:84:DD:56"
    # params.ip_address = "192.168.137.108"
    params.serial_number = "58a99b0107e64cd40ea5e6607882cbe2"
    params.board_id = board_id
    BoardShim.enable_dev_board_logger()
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    
    time.sleep(1)
    
    # for i in range(10):
    #     time.sleep(1)
    #     board.insert_marker(i + 1)
    
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    print(data)
    print(data.shape)

if __name__ == "__main__":
    main ()
