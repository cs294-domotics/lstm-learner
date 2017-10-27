import numpy as np

filename = "../data/twor2010"
desired_input_types = ['M', 'D'] #only motion and door sensors will be used in feature vector
desired_label_types = ['L'] #predicting light states

def main():
    with open(filename) as f:
		data = f.read().splitlines()
    num_events = len(data)
    # get list of all devices in dataset sorted into buckets by type
    device_buckets = get_devices(data)
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

# returns list of devices sorted into buckets by type
# eg, {'M': ['M001', 'M002'],
#      'D': ['D001']}
def get_devices(data):
    device_buckets = {}
    for line in data:
        if is_well_formed(line):
            device = get_device(line)
            device_type = get_device_type(device)
            if device_type not in device_buckets:
                device_buckets[device_type] = [device]
            else:
                if device not in device_buckets[device_type]:
                    device_buckets[device_type].append(device)
    return device_buckets

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

main()
