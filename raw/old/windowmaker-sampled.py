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
import random

WINDOW_TIME_SECS = 30
INTERVAL_SECS = 1
WINDOW_SIZE = int(WINDOW_TIME_SECS/INTERVAL_SECS)

LOAD_FOLDER = "stateful_data/"
#FILENAME = "twor2010_1s_one_day_train"
#FILENAME = "twor2010_1s_one_week_train"
#FILENAME = "twor2010_1s_one_week_test"
#FILENAME = "twor2010_1s_two_weeks_train"
FILENAME = "twor2010_1s_two_weeks_test"
#FILENAME = "stupid_simple_1s_full"
#FILENAME = "twor2010_1s_full"
LOAD_FILENAME = LOAD_FOLDER + FILENAME
SAVE_FOLDER = "build/sampled/"
SAVE_FILENAME = SAVE_FOLDER + FILENAME + "_" + str(WINDOW_TIME_SECS) + "s_window.h5"
INPUT_GROUP_NAME = 'input_matrices'
OUTPUT_GROUP_NAME = 'output_matrices'



def main():
    print("loading data...")
    lines = load_data(LOAD_FILENAME)
    # strip off the column labels
    #column_labels = lines.pop(0)
    num_lights = 11#get_num_light(column_labels)
    print("found {} lights...".format(num_lights))
    num_samples = len(lines)
    print("sampling data...")
    transition_indices = get_transition_indices(lines, num_lights, WINDOW_SIZE)
    print("found {} light transitions...".format(len(transition_indices)))
    other_indices = get_random_indices(lines, transition_indices, WINDOW_SIZE)
    print("sampled {} non-transitions...".format(len(other_indices)))
    all_indices = transition_indices + other_indices
    all_indices.sort()
    #print(all_indices)
    # make windows
    num_windows = len(all_indices)
    print("splitting data into {} {}s sampled windows ({} light transitions and {} non-transitions)...".format(num_windows, WINDOW_TIME_SECS, len(transition_indices), len(other_indices)))
    inputs = []
    outputs = []
    file_handle = tables.open_file(SAVE_FILENAME, mode='w')
    update_chunk = 2000
    num_processed = 0
    for output_window_end in all_indices:
        input_window, output_window = get_window(lines, output_window_end, WINDOW_SIZE)
        inputs.append(input_window)
        outputs.append(output_window)
        if num_processed >= update_chunk and num_processed % update_chunk == 0:
            print("{} processed of {} ({}%)".format(num_processed, num_windows, round(num_processed/num_windows*100, 2)))
            save_windows(inputs, file_handle, INPUT_GROUP_NAME)
            save_windows(outputs, file_handle, OUTPUT_GROUP_NAME)
            inputs = []
            outputs = []
        num_processed += 1
    save_windows(inputs, file_handle, INPUT_GROUP_NAME)
    save_windows(outputs, file_handle, OUTPUT_GROUP_NAME)
    file_handle.close()
    print("saved windows to " + SAVE_FILENAME)

def get_num_light(column_labels):
    fields = column_labels.split()
    num_lights = 0
    for field in fields:
        if field[0] == 'L':
            num_lights += 1
    return num_lights

def load_data(filename):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    return lines

def get_transition_indices(lines, num_lights, window_size):
    transition_indices = []
    s2_index = window_size
    num_processed = 0
    num_lines = len(lines) - window_size
    update_chunk = 200000
    while s2_index < len(lines):
        s1_light_state = get_light_state(lines[s2_index-1], num_lights)
        s2_light_state = get_light_state(lines[s2_index], num_lights)
        if s1_light_state != s2_light_state:
            transition_indices.append(s2_index)
        s2_index += 1
        num_processed += 1
        if num_processed >= update_chunk and num_processed % update_chunk == 0:
            print("{} processed of {} ({}%)".format(num_processed, num_lines, round(num_processed/num_lines*100, 2)))
    return transition_indices

def get_random_indices(lines, transition_indices, window_size):
    print("exclusions")
    excluded_indices = []
    for index in transition_indices:
        for i in range(window_size+1):
            excluded_indices.append(index-i)
    print("samples")

    num_samples = len(transition_indices)
    random_indices = []
    for i in range(num_samples):
        sample = random.randint(window_size+1, len(lines)-1)
        while sample in excluded_indices:
            sample = random.randint(window_size+1, len(lines)-1)
        random_indices.append(sample)
        for i in range(window_size+1):
            excluded_indices.append(sample-i)
    return random_indices

def get_light_state(line, num_lights):
    fields = line.split()
    return fields[(-1*num_lights):]

def get_window(lines, output_window_end, window_size):
    start_index = output_window_end - window_size
    end_index = output_window_end + 1
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
