#this is trying to predict a single light

import numpy as np
from hmmlearn import hmm
from copy import copy, deepcopy
from sklearn.preprocessing import normalize
import datetime
from time import time
import random
import os
import pprint


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

pp = pprint.PrettyPrinter(4)

def main():
    print(f'Loading data...')
    _, input_device_buckets, label_device_buckets, first_timestamps = load_data(master_file)
    train_data, _, _, _ = load_data(train_filename)
    test_data,  _, _, _ = load_data(test_filename)
    label_devices = flatten_buckets(label_device_buckets)
    _, _, event_labels = initialize_vector(input_device_buckets)
    label_invert = invert_dict(event_labels) 
    n_events = len(event_labels)
    train_vector = generate_sample_vector(train_data, event_labels) 
    to_model_space, from_model_space, filter_unseen, ms_indices = get_model_space(train_vector)
    # NOTE :: We filter out events in the test data that don't occour in the
    #        input data here, because our HMM chokes on symbols it hasn't seen
    #        during training. 
    test_vector = filter_unseen(generate_sample_vector(test_data, event_labels))
    ms_train_vector = to_model_space(train_vector)
    ms_test_vector = to_model_space(test_vector)
    ms_full_vector = np.append(ms_train_vector, ms_test_vector, 0) 
    ms_train_samples = ms_train_vector.shape[0]
    ms_test_samples = ms_test_vector.shape[0]
    ms_full_samples = ms_full_vector.shape[0]
    train_labels = to_label_list(train_vector, label_invert)
    test_labels = to_label_list(test_vector, label_invert)


    def run_training(num_states, max_iter=1000, conv_thresh=0.01):
        """
        This function will train an HMM with some number of states, and
        then verify it.

        We define this here because it needs all the earlier variables
        as context, but they don't need to be recalculated for each training
        operation we do. 
        """

        print(f'Training {num_states} state hmm with \'{train_filename}\':')        
        model = train_hmm(ms_train_vector, ms_indices, num_states,
                          max_iter, conv_thresh)
        print(f'  calculating predicted symbols...')
        # this is probability distribution over states after the HMM sees each
        # symbol 
        posteriors = model.predict_proba(ms_full_vector)
        # we insert the start condition, and since each posterior is the
        # prior for the next symbols, everything aligns
        priors = np.insert(posteriors, 0, model.startprob_, 0) 
        # Then we multiply each prior probability with the emission
        # matrix to get the predicted probabilities over each emitted
        # symbol
        emission_probs = np.matmul(priors, model.emissionprob_)
        # Then we take the symbol with the maximal probability at each step
        # so that we have the most likely outcome. 
        ms_predictions = np.argmax(emission_probs, axis=1).reshape(-1,1)
        # Convert that back into our original numerical labels
        predictions = from_model_space(ms_predictions)
        # split the predictions back out into their individual sequences. 
        train_predictions = predictions[0:ms_train_samples]
        test_predictions  = predictions[ms_train_samples:ms_train_samples+ms_test_samples]
        # convert those into their nice readable label forms. 
        train_prediction_labels = to_label_list(train_predictions, label_invert)
        test_prediction_labels  = to_label_list(test_predictions,  label_invert)
        print(f'  tabulating results...')
        train_acc, train_corr, train_count = get_accuracy(train_labels, train_prediction_labels)
        test_acc,  test_corr,  test_count  = get_accuracy(test_labels,  test_prediction_labels )
        print(f'    training data:')
        print(f'        accuracy: {train_acc}')
        print(f'        correct : {train_corr}')
        print(f'        total   : {train_count}')
        print(f'    test data:')
        print(f'        accuracy: {test_acc}')
        print(f'        correct : {test_corr}')
        print(f'        total   : {test_count}')
        return {'train': { 'accuracy' : train_acc,
                           'correct' : train_corr,
                           'total' : train_count},
                'test': { 'accuracy' : test_acc,
                           'correct' : test_corr,
                           'total' : test_count},
                'model': model}

    results = []
    for n in range(10,100,20):
        results.append(run_training(n))
        print(f'Results to date:')


    
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

def get_accuracy(data, preds):
    count = 0
    correct = 0
    for symb, pred in zip(data, preds):
        s = symb if symb[0] in desired_label_types else "OTHER"
        p = pred if pred[0] in desired_label_types else "OTHER"
        count += 1
        if s == p: correct += 1
    return (correct/count), correct, count

def to_label_list(data, invert_label):
    """
    Takes a numpy array in the usual space and converts it to a list of labels
    to make parsing and stuff easier. 
    """
    out = []
    for x in np.concatenate(data):
        out.append(invert_label[x])
    return out

def train_hmm(data, n_events, n_states, max_iter = 1000, conv_thresh = 0.01):
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
    print(f'  initializing hmm...') 
    model = hmm.MultinomialHMM(n_components = n_states,
                               init_params="",
                               n_iter=2,
                               tol= -999999999)
    model.emissionprob_ = normalize(np.random.rand(n_states, n_events))
    # We need to fit twice, because the model (for some reason) regresses
    # rather a lot on step 2 when we initialize from ... well any matrix
    # I've tried. From there it seems to work as expected so we set the number
    # of iterations and convergence threshold to somethig sane 
    model = model.fit(data, lengths=[data.shape[0]])
    print(f'  finished initialization...') 
    print(f'  starting training...') 
    model.n_iter = max_iter
    model.tol = conv_thresh
    model.verbose = True
    model = model.fit(data, lengths=[data.shape[0]])
    print(f'  finished training...') 
    return model

def get_model_space(data):
    """
    given a list of numbers, gets the set of unique values an makes them a
    consecutive series of integers, returns three functions to:
      - transform the original data into that consecutive set
      - transform from the consecutive space back into the original one
      - filter an array of all elements not in the original data.
    It also returns the number of states after redundant ones were eliminated
    """
    inverse_arr = np.unique(data)
    from_model_space_dict = make_inverse_dict(inverse_arr)
    to_model_space_dict = invert_dict(from_model_space_dict)
    fms_func = np.vectorize(lambda x: from_model_space_dict[x])
    tms_func = np.vectorize(lambda x: to_model_space_dict[x])
    def filter_unseen(dat):
        out = []
        for val in np.concatenate(dat):
            if val in to_model_space_dict:
                out.append(val)
        return np.array(out).reshape(-1,1) 
    return tms_func, fms_func, filter_unseen, len(from_model_space_dict)

def make_inverse_dict(inverse_array):
    """
    Take the inversion array that we get from np.unique an make it a dictionary
    like all the other translations we use. 
    """
    out = {}
    for ind, val in enumerate(inverse_array):
        out[ind] = val
    return out

def invert_dict(d):
    """
    Take a dictionary (assuming all values are unique) and invert it. 
    """
    out = {}
    for ind, val in d.items():
        out[val] = ind
    return out

def generate_sample_vector(data, event_labels):
    """
    NOTE :: heavily modified from the lstm preprocessor version

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
    NOTE: This is rather different from the version in the lstm preprocessor

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
