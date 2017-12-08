#K-19: The Windowmaker
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

LOAD_FOLDER = "stateful_data/"
FILENAME = "twor2010_1s_one_week_train"
#FILENAME = "twor2010_1s_one_week_test"
#FILENAME = "twor2010_1s_two_weeks_train"
#FILENAME = "twor2010_1s_two_weeks_test"
#FILENAME = "stupid_simple_1s_full"
LOAD_FILENAME = LOAD_FOLDER + FILENAME
SAVE_FOLDER = "build/non-overlapping"
SAVE_INPUT_FILENAME = SAVE_FOLDER + FILENAME + "_in.npy"
SAVE_OUTPUT_FILENAME = SAVE_FOLDER + FILENAME + "_out.npy"

WINDOW_TIME_SECS = 60
INTERVAL_SECS = 1
WINDOW_SIZE = int(WINDOW_TIME_SECS/INTERVAL_SECS)


def main():
    print("loading data...")
    lines = load_data(LOAD_FILENAME)
    # strip off the column labels
    column_labels = lines.pop(0)
    num_samples = len(lines)
    num_windows = int((num_samples-1)/WINDOW_SIZE)
    # make windows
    inputs = []
    outputs = []
    print("splitting data into " + str(WINDOW_TIME_SECS) + "s windows...")
    for window_num in range(num_windows):
        input_window, output_window = get_window(lines, window_num, WINDOW_SIZE)
        inputs.append(input_window)
        outputs.append(output_window)
    print("converting input windows to numpy arrays and saving...")
    save_windows(inputs, SAVE_INPUT_FILENAME)
    print("converting output windows to numpy arrays and saving...")
    save_windows(outputs, SAVE_OUTPUT_FILENAME)
    print("saved windows to " + SAVE_INPUT_FILENAME + " and " + SAVE_OUTPUT_FILENAME)

def load_data(filename):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    return lines

def get_window(lines, window_num, window_size):
    start_index = window_num * window_size
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

def save_windows(windows, filename):
    # build numpy matrices
    num_windows = len(windows)
    window_size = len(windows[0])
    num_features = len(windows[0][0])
    matrix = np.zeros((num_windows, window_size, num_features))
    # add the content
    for i in range(num_windows):
        for j in range(window_size):
            for k in range(num_features):
                try:
                    matrix[i][j][k] = windows[i][j][k]
                except ValueError:
                    print("ERROR:"),
                    print(windows[i])
    # write to file
    print(matrix)
    matrix.dump(open(filename, "wb"))

main()
