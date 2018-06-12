#this is trying to predict a single light

import numpy as np
from hmmlearn import hmm
from copy import copy, deepcopy
from sklearn.preprocessing import normalize
import datetime
from time import time
import random
import os


data_len = "_one_day"
#data_len = "_one_week"
#data_len = "_one_month"
train_file = data_len + "_train"
test_file  = data_len + "_test"
train_filename = "event_data/twor2010" + test_file
test_filename = "event_data/twor2010" + train_file
master_file = "../data/twor2010"

desired_events = {'M': ['OFF', 'ON'],
                  'D': ['CLOSE', 'OPEN'],
                  'L': ['OFF', 'ON'], #not interested in events like DIM:183 from L007
                  'A': ['START', 'END']}

desired_input_types = ['M', 'D', 'L'] # events from motion, door, and light sensors will be used in feature vector
desired_label_types = ['L'] #predicting the next light event

save_folder = "build/events/hmm/"


def main():
    _, input_device_buckets, label_device_buckets, first_timestamps = load_data(master_file)
    train_data, _, _, _ = load_data(train_filename)
    test_data,  _, _, _ = load_data(test_filename)
    label_devices = flatten_buckets(label_device_buckets)
    _, _, event_labels = initialize_vector(input_device_buckets)
    n_events = len(event_labels)
    print(event_labels)
    train_vector = generate_sample_vector(train_data, event_labels) 
    test_vector = generate_sample_vector(test_data, event_labels)
    print(n_events)
    model, model_space_inverse = train_hmm(train_vector, 20)
    # TODO :: - Take the training and test vectors and move them into model
    #          space, by inverting model_space_inverse.
    #        - Concat the model space train and test vectors
    #        - Generate the model priors for every step
    #        - multiply the model priors by the emission matrix for every
    #          step to get the probability for each next emitted element.
    #        - Take the index of the maximum value for each one
    #        - apply model_space_inverse to move everything back into
    #          event label space
    #        - iterate through predictions and results to properly bin correct
    #          and incorrect guesses, and get accuracy numbers. 

#    model = hmm.MultinomialHMM(n_components = 4, init_params="", verbose=True)
#    model.emissionprob_ = [[0.3, 0.3, 0.4],
#                           [0.2, 0.7, 0.1],
#                           [0.3, 0.1, 0.6],
#                           [0.1, 0.1 ,0.1]]
#
#    model.fit([[1],
#               [0],
#               [1],
#               [2]],
#              lengths = [4])
#    print(model.n_features)
#    print(model.emissionprob_)
#    print( model.sample(12))
# 3
# [[2.13928781e-01 6.46463204e-30 7.86071219e-01]
#  [5.44240045e-68 1.00000000e+00 1.62758405e-34]
#  [5.46941360e-01 2.21256534e-42 4.53058640e-01]
#  [1.69342389e-01 3.90903304e-23 8.30657611e-01]]
# (array([[1],
#        [2],
#        [1],
#        [2],
#        [1],
#        [0],
#        [1],
#        [0],
#        [1],
#        [2],
#        [1],
#        [0]]), array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]))


def train_hmm(data, n_states, max_iter = 1000, conv_thresh = 0.01):
    """
    data = the shape (len,1) sequence of symbols
    n_states = the number of hidden states we train with
    max_iter = the maximum number of training steps allowed.
    conv_thresh = when we see log odds improvements of below this amount then
                  stop. 

    returns:
      The model of the HMM trained on that sequence.
      The mapping of the 
    """

    # NOTE :: Okay, this stupid pile of crap requires that every event that might
    #        be in our list occour in some input sequence. So I'm going to
    #        get all the symbols that are in the input data, and create a
    #        sequence of all the non-relevant sym
    inverse, indices = np.unique(data, return_inverse=True)
    model = hmm.MultinomialHMM(n_components = n_states,
                               init_params="",
                               verbose=True,
                               n_iter=5,
                               tol= -999999999)
    model.emissionprob_ = normalize(np.random.rand(n_states, len(inverse)))
    data = indices.reshape(-1,1)
    # We need to fit twice, because the model (for some reason) regresses
    # rather a lot on step 2 when we initialize from ... well any matrix
    # I've tried. From there it seems to work as expected so we set the number
    # of iterations and convergence threshold to somethig sane 
    model = model.fit(data, lengths=[data.shape[0]])
    model.n_iter = max_iter
    model.tol = conv_thresh
    model = model.fit(data, lengths=[data.shape[0]])
    return model, inverse

def generate_sample_vector(data, event_labels):
    """
    given an input vector generates the sequence of symbols that serve
    as an output for a hidden markov machine. 
    """
    sequence = []
    for line in data:
        if is_well_formed(line):
            device = get_device(line)
            value = get_value(line)
            event = get_event(device, value)
            device_type = get_device_type(device)
            if device_type in desired_input_types:
                try:
                    sequence.append(event_labels[event])
                except KeyError:
                    pass #if it's a weird value (aka not ON/OFF OPEN/CLOSE, etc.), skip

    return np.array(sequence).reshape(-1,1)

def ensure_dir(dir_path):
    """
    Creates a directory if it doesn't exist
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_file_data(filename, gen):
    """
    Checks if a file exists, if it does load it. If it does not run the gen
    function, save the resulting data to that file, and return the same
    data.
    """
    pass


def load_data(filename):
    with open(filename) as f:
        data = f.read().splitlines()

    #print("removing all lines with devices in rooms outside of the living room...")
    #nearby_devices = ["M001", "M002", "M003", "M004", "M005", "M006", "M007", "M008", "M009", "M010", "M011", "M012", "M013", "M014", "M015", "M016", "M023" "D002", "L004", "L003", "L005"]
    #new_data = []
    #for line in data:
    #    d = get_device(line)
    #    if d in nearby_devices:
    #        new_data.append(line)
    #data = new_data

    # get list of all devices in dataset sorted into buckets by type
    device_buckets, first_timestamps = get_devices(data)
    # split into input and output devices
    input_device_buckets, label_device_buckets = filter_devices(device_buckets)
    return data, input_device_buckets, label_device_buckets, first_timestamps

def get_devices(data):
    """
    returns list of devices sorted into buckets by type
    eg, {'M': ['M001', 'M002'],
         'D': ['D001']}
    """
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

def is_well_formed(line):
    fields = line.split()
    return (len(fields) >= 4)

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

def encode(value):
    try:
        encoded_value = encoding[value]
    except KeyError:
        encoded_value = float(value) # This can throw value error
    return encoded_value

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

def flatten_buckets(buckets):
    flatten_buckets = []
    for bucket in buckets:
        flatten_buckets += buckets[bucket]
    return flatten_buckets

def initialize_vector(device_buckets):
    """
    figure out how many features the input vector will have
    devices and their possible event values according to type
    e.g., <m1_OFF, m1_ON, d1_CLOSED, d1_OPEN...>
    can't just record all the events we see, because we're not interested in
    all devices (don't care about P001) and we're not interest in all events
    (don't care about DIM:183 from L007, just ON/OFF).
    """
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

    #print(event_indices)

    # return the default vector with the correct number of features
    # and the mapping from each device to its index in the feature vector
    return initial_vector, device_events, event_indices

main()
