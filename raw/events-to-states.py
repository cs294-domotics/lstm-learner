# This preprocessor takes in a CASAS data file, which is in the form of
# timestamped events from single devices, and converts it into a file of
# stateful representations which look like they were polled at a fixed time
# interval. This file takes forever to run on twor2010.

import numpy as np
from copy import copy
import datetime
from time import time

LOAD_FOLDER = "../data/"
DATA_FILE = "twor2010"
#DATA_FILE = "stupid_simple"
FILENAME = LOAD_FOLDER + DATA_FILE
DESIRED_TYPES = ['M', 'D', 'L']
LIGHT_TYPE = 'L'
SAVE_FOLDER = "stateful_data/"

ENCODING = {'OFF': 0,
            'ON':  1,
            'CLOSE': 0,
            'OPEN': 1}

INDENT = "    "

INTERVAL_SECS = 1 #needs to divide evenly into WINDOW_TIME_SECS

def main():
    print("loading data...")
    data, device_buckets, first_timestamps = load_data(FILENAME)
    print("building stateful representations of events...")
    timestamps, stateful_data, device_indices = build_stateful_data(data, device_buckets)
    print("removing data that occurs before all devices have reported their state...")
    start_time = get_time_all_devices_seen(first_timestamps, DESIRED_TYPES)
    start_index = timestamps.index(start_time)
    stateful_data = stateful_data[start_index:]
    timestamps = timestamps[start_index:]
    print("expanding stateful representations of events into " + str(INTERVAL_SECS) + "s fixed-interval data...")
    save_file = generate_and_save_fixed_interval_data(timestamps, stateful_data, INTERVAL_SECS, device_indices)
    print("saved full fixed-interval data to " + save_file)

def get_filename(data_file, interval_secs, timeframe):
    name_template = "{}_{}s_{}"
    return SAVE_FOLDER + name_template.format(data_file, interval_secs, timeframe)

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
            if device_type in DESIRED_TYPES:
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
    print(INDENT + "total elapsed time: " + str(total_minutes) + " minutes")
    return timestamps, stateful_data, device_indices

def print_progress_report(samples_processed, num_samples, start_chunk_time, update_chunk):
    elapsed_minutes = round((time() - start_chunk_time)/60.0, 2)
    remaining_samples = num_samples - samples_processed
    remaining_chunks = remaining_samples / update_chunk
    remaining_minutes = round(remaining_chunks * elapsed_minutes, 2)
    print(INDENT + "processed " + str(samples_processed) + " out of " + str(num_samples))
    print(INDENT + str(elapsed_minutes) + " minutes elapsed since last update, " + str(remaining_minutes) + " estimated minutes remaining")

def get_time_all_devices_seen(first_timestamps, DESIRED_TYPES):
    # get first occurrences of label devices
    first_device_timestamps = filter_timestamps_by_device_type(first_timestamps, DESIRED_TYPES)
    # find the last first occurrence
    all_devices_seen_timestamp = get_last_timestamp(first_device_timestamps)
    return all_devices_seen_timestamp

def generate_and_save_fixed_interval_data(event_timestamps, stateful_event_data, interval_secs, device_indices):
    column_labels = "Timestamp " + " ".join(device_indices.keys()) + "\n"
    save_file = get_filename(DATA_FILE, INTERVAL_SECS, "full")
    f = open(save_file, "w")
    f.write(column_labels)

    # generate all interval timestamps
    print(INDENT + "generating interval timestamps...")
    interval_timestamps = []
    first_timestamp = event_timestamps[0]
    last_timestamp = event_timestamps[-1]
    duration_secs = last_timestamp - first_timestamp
    num_samples = int(duration_secs/interval_secs)
    for i in range(num_samples):
        interval_timestamps.append(first_timestamp + (i * interval_secs))

    print(INDENT + "generating interval states and writing to file...")
    # progress report info
    update_chunk = 200000 #samples
    samples_processed = 0
    start_time = time()
    start_chunk_time = start_time

    # initialize state w/first stateful event
    curr = 0
    current_state = stateful_event_data[curr]
    # initialize timestamp of next event
    next_event_timestamp = event_timestamps[curr+1]
    for interval_timestamp in interval_timestamps:
        while interval_timestamp > next_event_timestamp:
            curr += 1
            current_state = stateful_event_data[curr]
            next_event_timestamp = event_timestamps[curr+1]
        line_list = [interval_timestamp] + current_state
        f.write(" ".join([str(field) for field in line_list]) + "\n")
        # progress report info
        samples_processed += 1
        if samples_processed % update_chunk == 0:
            print_progress_report(samples_processed, num_samples, start_chunk_time, update_chunk)
            start_chunk_time = time()
    # final progress report
    total_minutes = round((time() - start_time)/60.0, 2)
    print(INDENT + "total elapsed time: " + str(total_minutes) + " minutes")
    f.close()
    return save_file

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

def filter_timestamps_by_device_type(timestamp_buckets, DESIRED_TYPES):
    timestamps = []
    for device in timestamp_buckets:
        if get_device_type(device) in DESIRED_TYPES:
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
        encoded_value = ENCODING[value]
    except KeyError:
        encoded_value = float(value) # This can throw value error
    return encoded_value

def initialize_vector(device_buckets):
    # figure out how many features the input vector will have
    num_features = 0
    device_indices = {}
    index = 0
    # guarantee that lights will be the last indices
    device_types = list(device_buckets.keys())
    device_types.pop(device_types.index(LIGHT_TYPE)) #remove 'L' from wherever
    device_types.append(LIGHT_TYPE) #put it back at the end
    for device_type in device_types:
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

################################################################################
#        BELOW FUNCTIONS NO LONGER USED DUE TO MEMORY INEFFICIENCY
################################################################################

def get_fixed_interval_data(event_timestamps, stateful_event_data, interval_secs):
    # generate all interval timestamps
    interval_timestamps = []
    first_timestamp = event_timestamps[0]
    last_timestamp = event_timestamps[-1]
    duration_secs = last_timestamp - first_timestamp
    num_samples = int(duration_secs/interval_secs)
    for i in range(num_samples):
        interval_timestamps.append(first_timestamp + (i * interval_secs))
    # initialize state w/first stateful event
    curr = 0
    current_state = stateful_event_data[curr]
    # initialize timestamp of next event
    next_event_timestamp = event_timestamps[curr+1]
    stateful_interval_data = []
    for interval_timestamp in interval_timestamps:
        while interval_timestamp > next_event_timestamp:
            curr += 1
            current_state = stateful_event_data[curr]
            next_event_timestamp = event_timestamps[curr+1]
        stateful_interval_data.append([interval_timestamp] + current_state)
    return stateful_interval_data

def save_data(device_indices, stateful_data, save_file):
    str_data = "Timestamp "
    str_data += " ".join(device_indices.keys()) + "\n"
    for state in stateful_data:
        str_data += " ".join([str(x) for x in state]) + "\n"
    str_data = str_data.rstrip()
    #print(str_data)
    f = open(save_file, "w")
    f.write(str_data)
    f.close()


main()
