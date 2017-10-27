import numpy as np

filename = "../data/simple"
desired_device_types = ['M', 'D'] #only motion and door sensors will be used in feature vector

def main():
    with open(filename) as f:
		data = f.read().splitlines()
    num_events = len(data)
    device_type_counts, device_type_buckets = get_devices(data)
    # for each timestep, assemble input vector
    input_vectors = []
    input_vector, device_indices = initialize_vector(device_type_counts, device_type_buckets, desired_device_types)
    for t in range(num_events):
        if is_well_formed(data[t]) and get_device_type(get_device(data[t])) in desired_device_types:
            #get new device/value
            device = get_device(data[t])
            value = get_value(data[t])
            #update input vector at device index to reflect newest value
            input_vector[device_indices[device]] = encode(value)
            #add input vector to input vectors
            input_vectors.append(input_vector)
            print(input_vector)

def get_devices(data):
    device_type_buckets = {}
    for line in data:
        if is_well_formed(line):
            device = get_device(line)
            device_type = get_device_type(device)
            if device_type not in device_type_buckets:
                device_type_buckets[device_type] = [device]
            else:
                if device not in device_type_buckets[device_type]:
                    device_type_buckets[device_type].append(device)
    device_type_counts = {}
    for device_type in device_type_buckets:
        device_type_counts[device_type] = len(device_type_buckets[device_type])
    return device_type_counts, device_type_buckets

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
    return encoding[value]

def initialize_vector(device_type_counts, device_type_buckets, desired_device_types):
    # figure out how many features the input vector will have
    num_features = 0
    for device_type in device_type_counts:
        if device_type in desired_device_types:
            num_features += device_type_counts[device_type]

    # assign an index to each device
    devices = []
    device_indices = {}
    for device_type in device_type_buckets:
        if device_type in desired_device_types:
            devices += device_type_buckets[device_type]
    i = 0
    for d in devices:
        device_indices[d] = i
        i+=1

    # create a default input vector
    initial_vector = []
    for i in range(num_features):
        initial_vector.append(0)

    # return the default vector with the correct number of features
    # and the mapping from each device to its index in the feature vector
    return initial_vector, device_indices

main()
