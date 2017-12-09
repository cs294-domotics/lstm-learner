from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop
from keras.utils.io_utils import HDF5Matrix
import numpy as np

calculate_transition_acc = True
calculate_real_world = False

folder = "build/overlapping/"
train_filename = folder + "twor2010_1s_two_weeks_train_15s_window.h5"
test_filename = folder + "twor2010_1s_two_weeks_test_15s_window.h5"
input_group = '/input_matrices'
output_group = '/output_matrices'

#transition_files = [folder + "twor2010_1s_two_weeks_test_30s_window.h5"]



# train_input_filename = folder + "twor2010_1s_two_weeks_train_60s_window_in.npy"
# train_output_filename = folder + "twor2010_1s_two_weeks_train_60s_window_out.npy"
# test_input_filename = folder + "twor2010_1s_two_weeks_test_60s_window_in.npy"
# test_output_filename = folder + "twor2010_1s_two_weeks_test_60s_window_out.npy"

NUM_LIGHTS = 11

def main():

    print("loading data...")
    # Generate training data
    # dimensions are batch_size, timesteps, data_sim
    #x_train = np.load(train_input_filename)
    #y_train = np.load(train_output_filename)
    x_train = HDF5Matrix(train_filename, input_group)
    y_train = HDF5Matrix(train_filename, output_group)

    # Generate validation data
    # dimensions are batch_size, timesteps, data_sim
    #x_val = np.load(test_input_filename)
    #y_val = np.load(test_output_filename)
    x_val = HDF5Matrix(test_filename, input_group)
    y_val = HDF5Matrix(test_filename, output_group)

    num_samples = len(x_train)
    timesteps = len(x_train[0])
    data_dim = len(x_train[0][0])
    num_outputs = data_dim

    print("checking data shapes...")

    if len(x_train) != len(y_train):
        print("uh oh...input and output for training are different sizes.")
    if len(x_val) != len(y_val):
        print("uh oh...input and output for testing are different sizes.")

    print("building model...")

    #num_hidden_states = 128
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(num_outputs, return_sequences=True, input_shape=(timesteps, data_dim)))
    #model.add(Dense(num_outputs, activation='tanh'))

    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop(lr=0.01),
                  metrics=['mae', 'categorical_accuracy'])

    print(model.summary())

    print("training and testing model...")
    model.fit(x_train, y_train,
            batch_size=256, epochs=3, #it kind of seems converged after 3...
            shuffle='batch')#,
            #validation_data=(x_val, y_val))


    if calculate_transition_acc == True:
        print("calculating transition accuracy...")
        print("selecting light transitions from test data...")
        transitional_x, transitional_y = get_transitional_samples(x_val, y_val)
        print("testing on {} light transitions...".format(len(transitional_x)))

        # for row in transitional_x[0]:
        #     print(row[-11:])
        # print('')
        # for row in transitional_y[0]:
        #     print(row[-11:]
        correct_transitions = []
        transition_counts = []
        correct_no_change = []
        no_change_counts = []
        for i in range(NUM_LIGHTS):
            correct_transitions.append(0)
            transition_counts.append(0)
            correct_no_change.append(0)
            no_change_counts.append(0)
        num_val = len(transitional_x)
        update_chunk = 200000
        for i in range(num_val):
            transition_mask, correctness_vector = get_light_accuracies(transitional_x[i], transitional_y[i], model.predict(np.reshape(transitional_x[i], (-1, timesteps, data_dim))))
            for (light_index, val) in enumerate(correctness_vector):
                if transition_mask[light_index] == True:
                    correct_transitions[light_index] += val # 1 if correct, 0 if not
                    transition_counts[light_index] += 1
                else:
                    correct_no_change[light_index] += val # 1 if correct, 0 if not
                    no_change_counts[light_index] += 1
            if i >= update_chunk and i % update_chunk == 0:
                print("{} processed of {} ({}%)".format(i, num_val, round(i/num_val*100, 2)))

        correct_transitions_mean = []
        correct_no_changes_mean = []
        for i in range(NUM_LIGHTS):
            if transition_counts[i] != 0:
                correct_transitions_mean.append(round(correct_transitions[i]/transition_counts[i]*100, 2))
            else:
                correct_transitions_mean.append("None")
            if no_change_counts[i] != 0:
                correct_no_changes_mean.append(round(correct_no_change[i]/no_change_counts[i]*100, 2))
            else:
                correct_no_changes_mean.append("None")
        sum_correct_transition_means = 0
        count_correct_transition_means = 0
        for mean in correct_transitions_mean:
            if mean != "None":
                sum_correct_transition_means += mean
                count_correct_transition_means += 1
        sum_correct_no_change_means = 0
        count_correct_no_change_means = 0
        for mean in correct_no_changes_mean:
            if mean != "None":
                sum_correct_no_change_means += mean
                count_correct_no_change_means += 1
        overall_transition_mean = sum_correct_transition_means/count_correct_no_change_means
        overall_no_change_mean = sum_correct_no_change_means/count_correct_no_change_means
        # omg no std dev?

        print("Transition Accuracy:")
        print("Overall mean accuracy on transitions: " + str(round(overall_transition_mean, 2)))
        for i in range(NUM_LIGHTS):
            print("    {}: {} ({}/{})".format(i, correct_transitions_mean[i], correct_transitions[i], transition_counts[i]))
        print("Overall mean accuracy on no-changes: " + str(round(overall_no_change_mean, 2)))
        for i in range(NUM_LIGHTS):
            print("    {}: {} ({}/{})".format(i, correct_no_changes_mean[i], correct_no_change[i], no_change_counts[i]))

    if calculate_real_world == True:
        print("\n***************\n")
        print("Calculating real-world performance...")
        # just do it on one month of overlapping data
        # initialize LSTM light state guess history with first light state ground truth history
        # for each input, feed in the historical sensor states + LSTM guess for light states
        # get LSTM guess, turn into ones and zeros.
        # extract light state guess
        # update light state guess history with latest
        # calculate accuracy compared to ground truth light state
        # (for each light, the LSTM agreed with ground truth X% of the time + std dev, and overall + std dev)





# returns those test samples where a light changes in the next state
def get_transitional_samples(x_val, y_val):
    transitional_x = []
    transitional_y = []
    num_samples = len(x_val)
    update_chunk = 20000
    for sample_num in range(num_samples):
        s1 = x_val[sample_num][-1]
        s2 = y_val[sample_num][-1]
        s1_lights = s1[(-1*NUM_LIGHTS):]
        s2_lights = s2[(-1*NUM_LIGHTS):]
        light_change = False
        for light_num in range(len(s1_lights)):
            if s1_lights[light_num] != s2_lights[light_num]:
                light_change = True
        if light_change:
            transitional_x.append(x_val[sample_num])
            transitional_y.append(y_val[sample_num])
        if sample_num >= update_chunk and sample_num % update_chunk == 0:
            print("{} processed of {} ({}%)".format(sample_num, num_samples, round(sample_num/num_samples*100, 2)))
    return transitional_x, transitional_y

# light accuracies is a list of tuples. Each index is for the light, the tuple there is the mean, std_dev.
def get_light_accuracies(x_true, y_true, y_pred):
    transition_mask = []
    correctness_vector = []
    for i in range(NUM_LIGHTS):
        transition_mask.append(False)
        correctness_vector.append(0)
    s1 = x_true[-1]
    s2 = y_true[-1]
    s2_pred = y_pred[0][-1]
    s1_lights = s1[(-1*NUM_LIGHTS):]
    s2_lights = s2[(-1*NUM_LIGHTS):]
    s2_lights_pred = s2_pred[(-1*NUM_LIGHTS):]
    # print("")
    # print(s1_lights)
    # print(s2_lights)
    # print(s2_lights_pred)
    # print("")
    for light_num in range(NUM_LIGHTS):
        if s1_lights[light_num] != s2_lights[light_num]:
            transition_mask[light_num] = True
        if s2_lights[light_num] == 1 and s2_lights_pred[light_num] >= 0.5:
            correctness_vector[light_num] = 1
        elif s2_lights[light_num] == 0 and s2_lights_pred[light_num] < 0.5:
            correctness_vector[light_num] = 1
    return transition_mask, correctness_vector

main()
