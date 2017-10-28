import numpy as np

filename = "../data/twor2010"
desired_input_types = ['M', 'D'] #only motion and door sensors will be used in feature vector
desired_label_types = ['L'] #predicting light states

def main():
    with open(filename) as f:
		data = f.read().splitlines()
    num_events = len(data)
    # get list of all devices in dataset sorted into buckets by type
    device_buckets, first_timestamps = get_devices(data)
    # split into input and output devices
    input_device_buckets, label_device_buckets = filter_devices(device_buckets)
    # initialize input and label vectors
    input_vectors = []
    label_vectors = []
    input_vector, input_device_indices = initialize_vector(input_device_buckets)
    label_vector, label_device_indices = initialize_vector(label_device_buckets)
    # for each timestep, update either input vector or label vector or neither (if a device type we're ignoring)
    for line in data:
        if is_well_formed(line):
            device = get_device(line)
            value = get_value(line)
            timestamp = get_timestamp(line) ############# AHHHHHHHHHH
            device_type = get_device_type(device)
            if device_type in desired_input_types or device_type in desired_label_types:
                if device_type in desired_input_types:
                    input_vector[input_device_indices[device]] = encode(value)
                elif device_type in desired_label_types:
                    label_vector[label_device_indices[device]] = encode(value)
                input_vectors.append(input_vector)
                label_vectors.append(label_vector)
                print(input_vector),
                print(label_vector)
    # Now we need to trim off the start of the dataset.
    # We need to have seen the state of all the desired device states
    # before we start training the learner to predict those states.
    # So we should find the timestep where we have seen all the desired label
    # devices report their state at least once, and remove everything before

    # get first occurrences of label devices
    first_label_timestamps = filter_timestamps_by_device_type(first_timestamps)
    # find the last first occurrence
    all_labels_seen_timestamp = get_last_timestamp(first_label_timestamps)
    # remove the stuff before the last first occurrence
    # input_vectors, label_vectors = trim(input_vectors, label_vectors, all_labels_seen_timestamp)

    # write to file
    #feature_matrix = np.zeros((num_samples, num_features))
	#label_vector = np.zeros(num_samples)
    #feature_matrix.dump(open("features.npy", "wb"))
	#label_vector.dump(open("labels.npy", "wb"))

def filter_timestamps_by_device_type(timestamp_buckets):
    timestamps = []
    for device in timestamp_buckets:
        if get_device_type(device) in desired_label_types:
            timestamps.append(timestamp_buckets[device])
    return timestamps

def get_last_timestamp(timestamps):
    timestamps.sort()
    return timestamps[-1]

# returns list of devices sorted into buckets by type
# eg, {'M': ['M001', 'M002'],
#      'D': ['D001']}
def get_devices(data):
    device_type_buckets = {}
    device_first_occurrences = {}
    for line in lines:
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

def filter_devices(device_buckets):
    input_device_buckets = {}
    label_device_buckets = {}
    for device_type in device_buckets:
        if device_type in desired_input_types:
            input_device_buckets[device_type] = device_buckets[device_type]
        elif device_type in desired_label_types:
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

def encode(value):
    encoding = {'OFF': 0,
                'ON':  1,
                'CLOSE': 0,
                'OPEN': 1}
    try:
        encoded_value = encoding[value]
    except KeyError:
        try:
            encoded_value = float(value)
        except ValueError:
            encoded_value = value
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

# first_label_timestamps = get_first_timestamps(label_device_buckets)
def get_first_timestamps(label_device_buckets):

def get_timestamp(line):
    fields = line.split()
    return fields[0] + ' ' + fields[1]

main()
