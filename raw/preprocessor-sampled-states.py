# This preprocessor takes in a CASAS data file and produces feature vectors and
# label vectors. Each feature vector is a snapshot of all the motion/door sensors
# at time t, and the label vector is a snapshot of all the lights at time t.
# For example, each timestamp in the original file will correspond to something
# like [0, 1, 0, 0, 1] => [0, 1], where two of the motion/door sensors have
# detected something and one of the lights is on.
# After assembling the complete state vectors for each timestep by holding
# values constant until a new update changes them, this preprocessor also trims
# the beginning of the dataset so that the final data only starts when all label
# devices have been seen. This is so that the learner is not training on
# unknown/incorrect labels.

import numpy as np
from copy import copy
import datetime
from time import time

filename = "../data/twor2010"
#filename = "../data/stupid_simple"
desired_types = ['M', 'D', 'L']
features_save_filename = "trajectory_features.npy"
labels_save_filename = "trajectory_labels.npy"
save_folder = "build/"

encoding = {'OFF': 0,
            'ON':  1,
            'CLOSE': 0,
            'OPEN': 1}

indent = "    "

ONE_DAY_SECS = 60*60*24
ONE_WEEK_SECS = ONE_DAY_SECS * 7
TWO_WEEK_SECS = ONE_WEEK_SECS * 2
ONE_MONTH_SECS = ONE_DAY_SECS * 30

WINDOW_TIME_SECS = 60 #5 * 60 #previous five minutes
SAMPLE_INTERVAL_SECS = 5 #needs to divide evenly into WINDOW_TIME_SECS
NUM_SAMPLES_PER_WINDOW = WINDOW_TIME_SECS / SAMPLE_INTERVAL_SECS
TIMEFRAMES = [ONE_DAY_SECS, ONE_WEEK_SECS, TWO_WEEK_SECS, ONE_MONTH_SECS]
TIMEFRAME_NAMES = {ONE_DAY_SECS: "one_day",
                   ONE_WEEK_SECS: "one_week",
                   TWO_WEEK_SECS: "two_weeks",
                   ONE_MONTH_SECS: "one_month"}

def main():
    print("loading data...")
    data, device_buckets, first_timestamps = load_data(filename)
    print("building stateful representations...")
    timestamps, stateful_data = build_stateful_data(data, device_buckets)
    print("removing data that occurs before all devices have reported their state...")
    start_time = get_time_all_devices_seen(first_timestamps, desired_types)
    start_index = timestamps.index(start_time)
    stateful_data = stateful_data[start_index:]
    timestamps = timestamps[start_index:]
    print("sampling and building trajectory windows...")
    for timeframe in TIMEFRAMES:
        training_points, testing_points = sample_stateful_data(timestamps, stateful_data, timeframe)
        training_trajectory_windows, training_next_in_windows = build_trajectories(timestamps, stateful_data, training_points)
        testing_trajectory_windows, testing_next_in_windows = build_trajectories(timestamps, stateful_data, testing_points)
        print("saving vectors for " + TIMEFRAME_NAMES[timeframe] + "...")
        training_features_file = save_folder + "training_" + TIMEFRAME_NAMES[timeframe] + "_" + features_save_filename
        training_labels_file = save_folder + "training_" + TIMEFRAME_NAMES[timeframe] + "_" + labels_save_filename
        testing_features_file = save_folder + "testing_" + TIMEFRAME_NAMES[timeframe] + "_" + features_save_filename
        testing_labels_file = save_folder + "testing_" + TIMEFRAME_NAMES[timeframe] + "_" + labels_save_filename
        save_vectors(training_trajectory_windows, training_features_file)
        save_vectors(training_next_in_windows, training_labels_file)
        save_vectors(testing_trajectory_windows, testing_features_file)
        save_vectors(testing_next_in_windows, testing_labels_file)
        print("saved vectors to " + training_features_file + ", " +
                                    training_labels_file + ", " +
                                    testing_features_file + ", and " +
                                    testing_labels_file)

def load_data(filename):
    with open(filename) as f:
        data = f.read().splitlines()
    # get list of all devices in dataset sorted into buckets by type
    device_buckets, first_timestamps = get_devices(data)
    return data, device_buckets, first_timestamps

def build_stateful_data(data, device_buckets):
    # progress report info
    update_chunk = 200000 #samples
    num_samples = len(data)
    samples_processed = 0
    start_time = time()
    start_chunk_time = start_time

    # build stateful data
    stateful_data = []
    timestamps = []
    state_vector, device_indices = initialize_vector(device_buckets)
    print(device_indices)
    for line in data:
        if is_well_formed(line):
            device = get_device(line)
            value = get_value(line)
            timestamp = get_timestamp(line)
            device_type = get_device_type(device)
            if device_type in desired_types:
                try:
                    state_vector[device_indices[device]] = encode(value)
                    stateful_data.append(copy(state_vector))
                    timestamps.append(timestamp)
                except ValueError: #if it's a weird value (aka not ON/OFF OPEN/CLOSE, or a number), skip
                    pass

        # progress report info
        samples_processed += 1
        if samples_processed % update_chunk == 0:
            print_progress_report(samples_processed, num_samples, start_chunk_time, update_chunk)
            start_chunk_time = time()
    # final progress report
    total_minutes = round((time() - start_time)/60.0, 2)
    print(indent + "total elapsed time: " + str(total_minutes) + " minutes")
    return timestamps, stateful_data

def print_progress_report(samples_processed, num_samples, start_chunk_time, update_chunk):
    elapsed_minutes = round((time() - start_chunk_time)/60.0, 2)
    remaining_samples = num_samples - samples_processed
    remaining_chunks = remaining_samples / update_chunk
    remaining_minutes = round(remaining_chunks * elapsed_minutes, 2)
    print(indent + "processed " + str(samples_processed) + " out of " + str(num_samples))
    print(indent + str(elapsed_minutes) + " minutes elapsed since last update, " + str(remaining_minutes) + " estimated minutes remaining")

def get_time_all_devices_seen(first_timestamps, desired_types):
    # get first occurrences of label devices
    first_device_timestamps = filter_timestamps_by_device_type(first_timestamps, desired_types)
    # find the last first occurrence
    all_devices_seen_timestamp = get_last_timestamp(first_device_timestamps)
    return all_devices_seen_timestamp

def sample_stateful_data(timestamps, stateful_data, timeframe):
    # get the timeframe of stateful data we're interested in
    # for both testing time and training timeframe:
        # figure out candidate timestamps for transitions
        # figure out candidate timestamps for stateful times
        # figure out the num_samples (the min of the number of candidates)
        # sample from the candidate transitions/stateful times to get an equal number of each
        # combine points in order
    return training_points, testing_points # timestamps of where points are in stateful_data


def build_trajectories(timestamps, stateful_data, points_of_interest):
    trajectory_windows = []
    next_in_windows = []

    return trajectory_windows, next_in_windows


def build_trajectory(timestamps, stateful_data, timestamp):
    # get the next state after the trajectory window
    next_state = stateful_data[index]
    next_state_timestamp = timestamps[index]
    # get the starting index of the initial state for the trajectory
    # initial state is the one before the window starts
    window_start_timestamp = next_state_timestamp - WINDOW_TIME_SECS
    #print(str(window_start_timestamp) + "\n")
    i = index
    while timestamps[i] >= window_start_timestamp:
        i = i - 1
        #print(timestamps[i])
    # get the states in the window + one before (the initial state)
    previous_states = stateful_data[i:index]
    previous_states_timestamps = timestamps[i:index]
    #print(previous_states)
    #print(previous_states_timestamps)
    # construct the trajectory as though the state were being polled at
    # a regular interval during that window
    initial_state = previous_states[0]
    trajectory = []
    poll_timestamps = generate_poll_timestamps(window_start_timestamp, next_state_timestamp)
    #print(poll_timestamps)
    #print(len(poll_timestamps))
    for poll_timestamp in poll_timestamps:
        # get the state vector that happened most recently at poll time
        preceding_state_index = get_index_of_preceding_number(poll_timestamp, previous_states_timestamps)
        #print(preceding_state_index)
        trajectory.append(previous_states[preceding_state_index])
    trajectory_windows.append(trajectory)
    next_in_windows.append(next_state)

    return trajectory_windows, next_in_windows

def get_index_of_preceding_number(n, sorted_number_list):
    i = len(sorted_number_list) - 1
    while n < sorted_number_list[i]:
        i = i - 1
    return i

def generate_poll_timestamps(window_start_timestamp, next_state_timestamp):
    duration_secs = next_state_timestamp - window_start_timestamp
    num_samples = int(duration_secs / SAMPLE_INTERVAL_SECS)
    poll_timestamps = []
    for i in range(num_samples):
        poll_timestamps.append(window_start_timestamp + SAMPLE_INTERVAL_SECS * i)
    return poll_timestamps

def get_windows(indices, trajectory_windows, window_size):
    input_samples = []
    label_samples = []
    for index in indices:
        label_vector = label_vectors[index]
        input_window = input_vectors[index+1-window_size:index+1]
        label_samples.append(label_vector)
        input_samples.append(input_window)
    return input_samples, label_samples

def save_vectors(vectors, filename):
    # build numpy matrices
    num_samples = len(vectors)
    num_features = len(vectors[0]) # semantically, num_outputs for the label vector
    matrix = np.zeros((num_samples, num_features))
    # add the content
    for i in range(num_samples):
        for j in range(num_features):
            try:
                matrix[i][j] = vectors[i][j]
            except ValueError:
                print("ERROR:"),
                print(vectors[i])
    # write to file
    print(matrix)
    matrix.dump(open(filename, "wb"))

# returns list of devices sorted into buckets by type
# eg, {'M': ['M001', 'M002'],
#      'D': ['D001']}
def get_devices(data):
    device_type_buckets = {}
    device_first_occurrences = {}
    for line in data:
        if is_well_formed(line):
            device = get_device(line)
            device_type = get_device_type(device)
            if device_type not in device_type_buckets:
                device_type_buckets[device_type] = [device]
                device_first_occurrences[device] = get_timestamp(line)
            else:
                if device not in device_type_buckets[device_type]:
                    device_type_buckets[device_type].append(device)
                    device_first_occurrences[device] = get_timestamp(line)
    return device_type_buckets, device_first_occurrences

def filter_timestamps_by_device_type(timestamp_buckets, desired_types):
    timestamps = []
    for device in timestamp_buckets:
        if get_device_type(device) in desired_types:
            timestamps.append(timestamp_buckets[device])
    return timestamps

def get_last_timestamp(timestamps):
    timestamps.sort()
    #print(timestamps)
    return timestamps[-1]

def is_well_formed(line):
    fields = line.split()
    return (len(fields) >= 4)

def get_device(line):
    return line.split()[2]

def get_device_type(device):
    return device[0]

def get_value(line):
    return line.split()[3]

def get_activity(line):
    return line.split()[4]

def encode(value):
    try:
        encoded_value = encoding[value]
    except KeyError:
        encoded_value = float(value) # This can throw value error
    return encoded_value

def initialize_vector(device_buckets):
    # figure out how many features the input vector will have
    num_features = 0
    device_indices = {}
    index = 0
    for device_type in device_buckets:
        num_features += len(device_buckets[device_type])
        # assign an index to each device
        for device in device_buckets[device_type]:
            device_indices[device] = index
            index += 1

    # create a default vector
    initial_vector = []
    for i in range(num_features):
        initial_vector.append(0)

    # return the default vector with the correct number of features
    # and the mapping from each device to its index in the feature vector
    return initial_vector, device_indices

def get_timestamp(line):
    fields = line.split()
    timestamp_str = fields[0] + ' ' + fields[1]
    try:
        timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f").timestamp()
    except ValueError:
        timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").timestamp()
    return timestamp

main()
