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
#filename = "../data/simple"
desired_types = ['M', 'D', 'L']
features_save_filename = "trajectory_features.npy"
labels_save_filename = "trajectory_labels.npy"

encoding = {'OFF': 0,
            'ON':  1,
            'CLOSE': 0,
            'OPEN': 1}

indent = "    "

WINDOW_TIME_SECS = 5 * 60 #previous five minutes
SAMPLE_INTERVAL_SECS = 0.5

def main():
    print("loading data...")
    data, device_buckets, first_timestamps = load_data(filename)
    print("building stateful representations...")
    timestamps, stateful_data = build_stateful_data(data, device_buckets)
    print("removing data that occurs before all devices have reported their state...")
    start_time = get_time_all_devices_seen(first_timestamps, desired_types)
    start_index = timestamps.index(start_time)
    stateful_data = stateful_data[start_index:]
    print("sampling and building trajectory windows...")
    trajectory_windows, next_in_windows = build_trajectories(timestamps, stateful_data)
    print("saving vectors...")
    save_vectors(trajectory_windows, features_save_filename)
    save_vectors(next_in_windows, labels_save_filename)
    print("saved vectors to " + features_save_filename + " and " + labels_save_filename)

def load_data(filename):
    with open(filename) as f:
        data = f.read().splitlines()
    # get list of all devices in dataset sorted into buckets by type
    device_buckets, first_timestamps = get_devices(data)
    return data, device_buckets, first_timestamps

def build_stateful_vectors(lines, device_buckets):
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
    for line in lines:
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

def build_trajectories(timestamps, stateful_data):
    trajectory_windows = []
    next_in_windows = []
    # STEP 1: pick indices of goal states to learn
    # OPTION 1: Good ones to target are situations where the light changes
    # There should probably be roughly equal numbers of examples where some light changes
    # And where no lights change
    # OPTION 2: maybe sample a random percent of them, so that the thing doesn't explode on me?
    # problem is that doesn't represent the deployment setup very well.
    # Whatever, let's just see if it works for now.

    # STEP 2:
    # get the stateful data from the preceding WINDOW_TIME_SECS (and one before to initialize)
    # then construct sample window with SAMPLE_INTERVAL_SECS number of state vectors
    # which change according to the timestamp info (basically, filling in gaps)
    for index in indices:
        next_state = stateful_data[index]
        next_state_timestamp = timestamps[index]
        window_start_timestamp = next_state_timestamp - WINDOW_TIME_SECS
        previous_states = ... #everything between timestamp of next_state-WINDOW_TIME_SECS
        previous_states_timestamps = 
        trajectory = []
        for state in previous_states:
    return trajectory_windows, next_in_windows

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
