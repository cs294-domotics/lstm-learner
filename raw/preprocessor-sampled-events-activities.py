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
from copy import copy, deepcopy
import datetime
from time import time
import random

#filetime = "_one_day_train"
#filetime = "_one_day_test"
#filetime = "_one_week_train"
#filetime = "_one_week_test"
#filetime = "_two_weeks_train"
#filetime = "_two_weeks_test"
#filetime = "_one_month_train"
filetime = "_one_month_test"
filename = "event_data/twor2010" + filetime
master_file = "../data/twor2010"

#filename = "../data/twor2010"
#filename = "../data/stupid_simple"
desired_input_types = ['M', 'D', 'L'] # events from motion, door, and light sensors will be used in feature vector
desired_label_types = ['L'] #predicting the next light event
features_save_filename = "features.npy"
labels_save_filename = "labels.npy"
save_folder = "build/events/activities/light_and_time/"
#save_folder = "build/events/activities/no_light_no_time/"

add_time = True
add_light_state = True

desired_events = {'M': ['OFF', 'ON'],
                  'D': ['CLOSE', 'OPEN'],
                  'L': ['OFF', 'ON'], #not interested in events like DIM:183 from L007
                  'A': ['START', 'END']}

indent = "    "

window_size = 40 # need to find best window size

def main():
    # get data and metadata
    print("loading data...")
    _, input_device_buckets, label_device_buckets, first_timestamps, activity_list = load_data(master_file)
    data, _, _, _, _ = load_data(filename)
    label_devices = flatten_buckets(label_device_buckets)
    # build the input/label vectors and keep track of timestamps
    print("building vector representation of data...")
    input_vectors, label_vectors, label_device_events, label_event_indices, timestamps, activity_indices = build_vectors(data, input_device_buckets, label_device_buckets, activity_list)
    # use timestamps to remove vectors that occur before we've heard from all label devices at least once
    print("aligning vector representation of data...")
    timestamps = timestamps[:-1]
    input_vectors = input_vectors[:-1] #off-by-one to align the events with the following light action
    label_vectors = label_vectors[1:]
    if add_time:
        print("adding time of day information...")
        # for the combined vectors, add in day/night info
        input_vectors = add_time_of_day(input_vectors, timestamps)
    print("generating label vectors for each device...") # one-hot vectors <turned_off, turned_on, no_change>
    device_label_vectors = {}
    for device in label_devices:
        if device == 'L005':
            device_label_vectors[device] = generate_device_label_vectors(label_device_events[device], label_event_indices, label_vectors)
        else:
            device_label_vectors[device] = label_vectors
    if add_light_state:
        print("adding state info to each device's feature vectors...")
    # for each light's vectors, insert the state information into the feature vectors
    device_feature_vectors = {}
    for device in label_devices:
        if device == 'L005':
            if add_light_state:
                device_feature_vectors[device] = generate_device_feature_vectors(input_vectors, device_label_vectors[device])
            else:
                device_feature_vectors[device] = input_vectors
        else:
            device_feature_vectors[device] = input_vectors
    print("sampling to generate balanced input for each class...")
    for device in label_devices:
        if device == 'L005':
            input_samples, label_samples = select_samples(device, device_feature_vectors[device], device_label_vectors[device], window_size)
            print("saving samples for " + str(device) + "...")
            feature_filename = save_folder + str(device) + "_" + str(window_size) + filetime + "_" + features_save_filename
            label_filename = save_folder + str(device) + "_" + str(window_size) + filetime + "_" + labels_save_filename
            save_vectors(input_samples, label_samples, feature_filename, label_filename)
            print("saved samples to " + feature_filename + " and " + label_filename)

def add_time_of_day(input_vectors, timestamps):
    time_of_day = 0 #night
    for (i, t) in enumerate(timestamps):
        # get hour of day
        date = datetime.datetime.fromtimestamp(t)
        #print(date.hour)
        if date.hour > 6 and date.hour < 18:
            time_of_day = 1 #day
        input_vectors[i] = input_vectors[i] + [date.hour/24.0]
    return input_vectors

def generate_device_feature_vectors(input_vectors, label_vectors):
    light_state = 0
    vectors = deepcopy(input_vectors)
    for (i, states) in enumerate(label_vectors):
        vectors[i] = vectors[i] + [light_state]
        if states[0] == 1:
            light_state = 0
        elif states[1] == 1:
            light_state = 1
    return vectors

def generate_device_label_vectors(device_events, event_indices, event_vectors):
    device_event_vectors = []
    device_event_indices = []
    for event in device_events:
        device_event_indices.append(event_indices[event])
    for event_vector in event_vectors:
        # initialize new device event vector
        device_event_vector = []
        for i in range(len(device_event_indices) + 1):
            device_event_vector.append(0)
        # fill in the change events
        for (i, index) in enumerate(device_event_indices):
            device_event_vector[i] = event_vector[index]
        # if no change, set nop field to 1
        if sum(device_event_vector) == 0.0:
            device_event_vector[-1] = 1
        device_event_vectors.append(device_event_vector)
    return device_event_vectors


def flatten_buckets(buckets):
    flatten_buckets = []
    for bucket in buckets:
        flatten_buckets += buckets[bucket]
    return flatten_buckets

def load_data(filename):
    with open(filename) as f:
        data = f.read().splitlines()

    #print("removing all lines with devices in rooms outside of the living room...")
    #nearby_devices = ["M001", "M002", "M003", "M004", "M005", "M006", "M007", "M008", "M009", "M010", "M011", "M012", "M013", "M014", "M015", "M016", "M023" "D002", "L004", "L003", "L005"]
    #new_data = []
    #for line in data:
#        d = get_device(line)
#        if d in nearby_devices:
#            new_data.append(line)
#    data = new_data

    # get list of all devices in dataset sorted into buckets by type
    device_buckets, device_first_timestamps, activity_list = get_devices_and_activities(data)
    # split into input and output devices
    input_device_buckets, label_device_buckets = filter_devices(device_buckets)
    return data, input_device_buckets, label_device_buckets, device_first_timestamps, activity_list

def build_vectors(data, input_device_buckets, label_device_buckets, activity_list):
    # initialize progress tracker
    num_samples = len(data)
    samples_processed = 0
    update_chunk = 200000 #samples
    start_time = time()
    start_chunk_time = start_time

    # initialize input and label vectors
    input_vectors = []
    label_vectors = []
    timestamps = []
    input_vector, input_device_events, input_event_indices, activity_indices = initialize_input_vector(input_device_buckets, activity_list)
    label_vector, label_device_events, label_event_indices = initialize_output_vector(label_device_buckets)
    state_vector = zero_vector(len(activity_list))
    print(activity_indices)
    print(input_event_indices)
    print(label_event_indices)
    # for each timestep, update either input vector or label vector or neither (if a device type we're ignoring)
    for line in data:
        input_vector = zero_vector(len(input_vector))
        input_vector[:len(activity_list)] = copy(state_vector)
        label_vector = zero_vector(len(label_vector))
        # put the current activity state into the vector
        if is_well_formed(line):
            device = get_device(line)
            value = get_value(line)
            event = get_event(device, value)
            timestamp = get_timestamp(line)
            device_type = get_device_type(device)
            activity = get_activity(line)
            activity_event = encode_activity_event(get_activity_event(line))
            if activity is not None and activity_event is not None:
                state_vector[activity_indices[activity]] = activity_event
            if device_type in desired_input_types or device_type in desired_label_types:
                try:
                    if device_type in desired_input_types:
                        input_vector[input_event_indices[event]] = 1
                    if device_type in desired_label_types:
                        label_vector[label_event_indices[event]] = 1
                    input_vectors.append(copy(input_vector))
                    label_vectors.append(copy(label_vector))
                    timestamps.append(timestamp)
                except KeyError:
                    pass #if it's a weird value (aka not ON/OFF OPEN/CLOSE, etc.), skip
                #print(input_vector),
                #print(label_vector)
        samples_processed += 1
        if samples_processed % update_chunk == 0:
            elapsed_minutes = round((time() - start_chunk_time)/60.0, 2)
            remaining_samples = num_samples - samples_processed
            remaining_chunks = remaining_samples / update_chunk
            remaining_minutes = round(remaining_chunks * elapsed_minutes, 2)
            print(indent + "processed " + str(samples_processed) + " out of " + str(num_samples))
            print(indent + str(elapsed_minutes) + " minutes elapsed since last update, " + str(remaining_minutes) + " estimated minutes remaining")
            start_chunk_time = time()
    total_minutes = round((time() - start_time)/60.0, 2)
    print(indent + "total elapsed time: " + str(total_minutes) + " minutes")
    return input_vectors, label_vectors, label_device_events, label_event_indices, timestamps, activity_indices

def encode_activity_event(activity_event):
    state = None
    if activity_event is not None:
        if activity_event.lower() == 'start' or activity_event.lower() == 'begin':
            state = 1
        elif activity_event.lower() == 'end':
            state = 0
    return state

def get_time_all_devices_seen(first_timestamps, desired_types):
    # get first occurrences of label devices
    first_device_timestamps = filter_timestamps_by_device_type(first_timestamps, desired_types)
    # find the last first occurrence
    all_devices_seen_timestamp = get_last_timestamp(first_device_timestamps)
    return all_devices_seen_timestamp

# returns about equal numbers of samples for each label
# each label will have window_size number of events running up to it, and then
# the label
def select_samples(device, input_vectors, device_label_vectors, window_size):
    if len(input_vectors) != len(device_label_vectors):
        print("PROBLEM: input vectors and label vectors for device " + str(device) + " are different lengths")

    # initialize the index buckets for each class/label
    class_indices = []
    num_classes = len(device_label_vectors[0])
    for c in range(num_classes):
        class_indices.append([])

    # get the indices for each class/label (aka the indices where the light turned on, turn off, did nothing))
    for i in range(window_size, len(device_label_vectors)):
        for class_index in range(num_classes):
            if device_label_vectors[i][class_index] == 1:
                class_indices[class_index].append(i)

    # choose indices of examples for each class
    # in practice, there are the same number of on and off labels, so just accept
    # all those indices without bothering
    class_sample_indices = class_indices[:-1]
    print("     Number of OFF samples: " + str(len(class_sample_indices[0]))), #sanity checks
    print("     Number of ON samples: " + str(len(class_sample_indices[1])))
    # choose random indices for the no-change events
    num_samples = len(class_sample_indices[0])
    no_change_indices = []
    for i in sorted(random.sample(range(window_size, len(class_indices[-1])), num_samples)):
        no_change_indices.append(class_indices[-1][i])
    class_sample_indices.append(no_change_indices)
    print("     Number of NO CHANGE samples: " + str(len(class_sample_indices[2])))

    # slice out the window of events before each index and pair it with the event
    indices = [index for sample_indices in class_sample_indices for index in sample_indices]
    input_samples, label_samples = get_windows(indices, input_vectors, device_label_vectors, window_size)

    return input_samples, label_samples

def get_windows(indices, input_vectors, label_vectors, window_size):
    input_samples = []
    label_samples = []
    for index in indices:
        label_vector = label_vectors[index]
        input_window = input_vectors[index+1-window_size:index+1]
        label_samples.append(label_vector)
        input_samples.append(input_window)
    return input_samples, label_samples


def save_vectors(input_samples, label_samples, input_filename, label_filename):
    if len(input_samples) != len(label_samples):
        print("PROBLEM: Different number of samples and labels")
    # build numpy matrices
    num_samples = len(input_samples)
    num_timesteps = len(input_samples[0])
    num_features = len(input_samples[0][0])
    num_classes = len(label_samples[0])
    input_matrix = np.zeros((num_samples, num_timesteps, num_features))
    label_vector = np.zeros((num_samples, num_classes))
    # add the content
    for i in range(num_samples):
        for j in range(num_timesteps):
            for k in range(num_features):
                try:
                    input_matrix[i][j][k] = input_samples[i][j][k]
                except ValueError:
                    print("ERROR creating numpy feature matrix")
        for n in range(num_classes):
            label_vector[i][n] = label_samples[i][n]
    # write to file
    print(input_matrix[:3])
    print(label_vector[:3])
    input_matrix.dump(open(input_filename, "wb"))
    label_vector.dump(open(label_filename, "wb"))

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

# returns list of devices sorted into buckets by type
# eg, {'M': ['M001', 'M002'],
#      'D': ['D001']}
def get_devices_and_activities(data):
    activity_list = []
    device_type_buckets = {}
    device_first_occurrences = {}
    for line in data:
        if is_well_formed(line):
            device = get_device(line)
            device_type = get_device_type(device)
            activity = get_activity(line)
            if device_type not in device_type_buckets:
                device_type_buckets[device_type] = [device]
                device_first_occurrences[device] = get_timestamp(line)
            else:
                if device not in device_type_buckets[device_type]:
                    device_type_buckets[device_type].append(device)
                    device_first_occurrences[device] = get_timestamp(line)
            if activity is not None and activity not in activity_list:
                activity_list.append(activity)
    return device_type_buckets, device_first_occurrences, activity_list

def filter_devices(device_buckets):
    input_device_buckets = {}
    label_device_buckets = {}
    for device_type in device_buckets:
        if device_type in desired_input_types:
            input_device_buckets[device_type] = device_buckets[device_type]
        if device_type in desired_label_types:
            label_device_buckets[device_type] = device_buckets[device_type]
    return input_device_buckets, label_device_buckets

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

def get_event(device, value):
    return str(device) + '_' + str(value)

def get_activity(line):
    activity = None
    fields = line.split()
    if len(fields) >= 6:
        activity = fields[4]
    return activity

def get_activity_event(line):
    activity_event = None
    fields = line.split()
    if len(fields) >= 6:
        activity_event = fields[5]
    return activity_event

def encode(value):
    try:
        encoded_value = encoding[value]
    except KeyError:
        encoded_value = float(value) # This can throw value error
    return encoded_value

def initialize_input_vector(device_buckets, activity_list):
    # figure out how many features the input vector will have
    num_features = 0
    activity_indices = {}
    event_indices = {}
    device_events = {}
    index = 0

    # put activities in front
    for activity in activity_list:
        num_features += 1
        activity_indices[activity] = index
        index += 1

    for device_type in device_buckets:
        for device in device_buckets[device_type]:
            device_events[device] = []
            for value in desired_events[device_type]:
                event = get_event(device, value)
                device_events[device].append(event)
                event_indices[event] = num_features
                num_features += 1

    # create a default vector
    initial_vector = zero_vector(num_features)
    return initial_vector, device_events, event_indices, activity_indices

def initialize_output_vector(device_buckets):
    # figure out how many features the input vector will have
    # devices and their possible event values according to type
    # e.g., <m1_OFF, m1_ON, d1_CLOSED, d1_OPEN...>
    # can't just record all the events we see, because we're not interested in
    # all devices (don't care about P001) and we're not interest in all events
    # (don't care about DIM:183 from L007, just ON/OFF).
    feature_count = 0
    event_indices = {}
    device_events = {}
    for device_type in device_buckets:
        for device in device_buckets[device_type]:
            device_events[device] = []
            for value in desired_events[device_type]:
                event = get_event(device, value)
                device_events[device].append(event)
                event_indices[event] = feature_count
                feature_count += 1

    # create a default vector
    initial_vector = zero_vector(feature_count)

    print(event_indices)

    # return the default vector with the correct number of features
    # and the mapping from each device to its index in the feature vector
    return initial_vector, device_events, event_indices

def zero_vector(length):
    vector = []
    for i in range(length):
        vector.append(0)
    return vector

def get_timestamp(line):
    fields = line.split()
    timestamp_str = fields[0] + ' ' + fields[1]
    try:
        timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f").timestamp()
    except ValueError:
        timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").timestamp()
    return timestamp

def test():
    with open(filename) as f:
        lines = f.read().splitlines()
    activities = {}
    for line in lines:
        fields = line.split()
        if len(fields) == 6:
            activity = fields[4]
            activity_event = fields[5]
            if activity not in activities:
                print(activity)
                activities[activity] = 1
            else:
                activities[activity] += 1
    print(activities)


#test()
main()
