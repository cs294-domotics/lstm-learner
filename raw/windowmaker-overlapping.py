# Fun fact: dumping large files doesn't work...

# K-19: The Windowmaker
# Takes in a fixed-interval stateful data file formatted like:
#
# Timestamp M050 M044 M045 L005
# 1251097367.069235 0 0 0 1
# 1251097368.069235 0 0 0 1
# 1251097369.069235 0 0 1 1
# ...
#
# and turns it into numpy arrays for feeding into an LSTM.

import numpy as np
from copy import copy
import tables

WINDOW_TIME_SECS = 1
INTERVAL_SECS = 1
WINDOW_SIZE = int(WINDOW_TIME_SECS/INTERVAL_SECS)

LOAD_FOLDER = "stateful_data/"
#FILENAME = "twor2010_1s_one_day_train"
#FILENAME = "twor2010_1s_one_day_test"
#FILENAME = "twor2010_1s_one_week_train"
#FILENAME = "twor2010_1s_one_week_test"
#FILENAME = "twor2010_1s_two_weeks_train"
FILENAME = "twor2010_1s_two_weeks_test"
#FILENAME = "twor2010_1s_one_month_train"
#FILENAME = "twor2010_1s_one_month_test"
#FILENAME = "stupid_simple_1s_full"
LOAD_FILENAME = LOAD_FOLDER + FILENAME
SAVE_FOLDER = "build/overlapping/"
SAVE_FILENAME = SAVE_FOLDER + FILENAME + "_" + str(WINDOW_TIME_SECS) + "s_window.h5"
INPUT_GROUP_NAME = 'input_matrices'
OUTPUT_GROUP_NAME = 'output_matrices'


def main():
    print("loading data...")
    lines = load_data(LOAD_FILENAME)
    # strip off the column labels
    column_labels = lines.pop(0)
    num_samples = len(lines)
    num_windows = len(lines) - WINDOW_SIZE
    # make windows
    inputs = []
    outputs = []
    print("splitting data into {} {}s windows...".format(num_windows, WINDOW_TIME_SECS))
    file_handle = tables.open_file(SAVE_FILENAME, mode='w')
    update_chunk = 20000
    for window_num in range(num_windows):
        input_window, output_window = get_window(lines, window_num, WINDOW_SIZE)
        inputs.append(input_window)
        outputs.append(output_window)
        if window_num >= update_chunk and window_num % update_chunk == 0:
            print("{} processed of {} ({}%)".format(window_num, num_windows, round(window_num/num_windows*100, 2)))
            save_windows(inputs, file_handle, INPUT_GROUP_NAME)
            save_windows(outputs, file_handle, OUTPUT_GROUP_NAME)
            inputs = []
            outputs = []
    save_windows(inputs, file_handle, INPUT_GROUP_NAME)
    save_windows(outputs, file_handle, OUTPUT_GROUP_NAME)
    file_handle.close()
    print("saved windows to " + SAVE_FILENAME)

def load_data(filename):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    return lines

def get_window(lines, window_num, window_size):
    start_index = window_num
    end_index = start_index + window_size + 1
    trajectory_lines = lines[start_index:end_index]
    trajectory = []
    for line in trajectory_lines:
        fields = line.split()
        state = fields[1:]
        trajectory.append([float(x) for x in state])
    input_window = trajectory[:-1]
    output_window = trajectory[1:]
    return input_window, output_window

def save_windows(windows, file_handle, group):
    num_windows = len(windows)
    window_size = len(windows[0])
    num_features = len(windows[0][0])

    try:
        matrices = file_handle.get_node("/" + group)
    except tables.exceptions.NoSuchNodeError:
        matrices = file_handle.create_earray(file_handle.root, group, tables.Atom.from_dtype(np.array(windows[0]).dtype), (0,window_size,num_features))

    # build numpy matrices
    matrix = np.zeros((window_size, num_features))
    # add the content
    for i in range(num_windows):
        matrix = np.zeros((window_size, num_features))
        for j in range(window_size):
            for k in range(num_features):
                matrix[j][k] = windows[i][j][k]
        matrices.append(matrix.reshape(-1, window_size, num_features))

main()
