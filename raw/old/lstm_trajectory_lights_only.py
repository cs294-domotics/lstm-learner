from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop
from keras.utils.io_utils import HDF5Matrix
import numpy as np

calculate_transition_acc = True
calculate_real_world = False

#folder = "build/sampled/"
#train_filename = folder + "twor2010_1s_two_weeks_train_30s_window.h5"
#test_filename = folder + "twor2010_1s_two_weeks_test_30s_window.h5"
#folder = "build/lights-only-overlapping/activities/"
folder = "build/lights-only-overlapping/downsampled/"
#folder = "build/lights-only-overlapping/"
train_filename = folder + "twor2010_1s_two_weeks_train_15s_window.h5"
test_filename = folder + "twor2010_1s_two_weeks_test_15s_window.h5"
input_group = '/input_matrices'
output_group = '/output_matrices'

transition_files = [folder + "twor2010_1s_two_weeks_test_15s_window.h5"]

# transition_files = [folder + "twor2010_1s_one_day_test_15s_window.h5",
#                     folder + "twor2010_1s_one_week_test_15s_window.h5",
#                     folder + "twor2010_1s_two_weeks_test_15s_window.h5",
#                     folder + "twor2010_1s_one_month_test_15s_window.h5"]

real_world_folder = "build/lights-only-overlapping/"
real_world_files = [real_world_folder + "twor2010_1s_two_weeks_test_15s_window.h5"]



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
    num_outputs = len(y_train[0])

    print("checking data shapes...")

    if len(x_train) != len(y_train):
        print("uh oh...input and output for training are different sizes.")
    if len(x_val) != len(y_val):
        print("uh oh...input and output for testing are different sizes.")

    print("building model...")

    num_hidden_states = 128#20
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(num_hidden_states, return_sequences=False, input_shape=(timesteps, data_dim)))
    model.add(Dense(num_outputs, activation='sigmoid'))

    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop(lr=0.01),
                  metrics=['mae'])

    print(model.summary())

    print("training and testing model...")
    model.fit(x_train, y_train,
            #batch_size=256, epochs=4, #it kind of seems converged after 3...
            batch_size = 4, epochs=9,
            shuffle='batch')

    if calculate_transition_acc == True:
        for t_filename in transition_files:
            x_val = HDF5Matrix(t_filename, input_group)
            y_val = HDF5Matrix(t_filename, output_group)
            print("\ncalculating transition accuracy...")
            print("selecting light transitions from {}...".format(t_filename))
            transitional_x, transitional_y, transition_masks = get_transitional_samples(x_val, y_val)
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
                correctness_vector = get_light_accuracies(transitional_y[i], model.predict(np.reshape(transitional_x[i], (-1, timesteps, data_dim))))
                transition_mask = transition_masks[i]
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

            filtered_correct_transitions_mean = []
            filtered_correct_no_changes_mean = []
            for i in range(NUM_LIGHTS):
                if correct_transitions_mean[i] != "None":
                    filtered_correct_transitions_mean.append(correct_transitions_mean[i])
                if correct_no_changes_mean[i] != "None":
                    filtered_correct_no_changes_mean.append(correct_no_changes_mean[i])

            filtered_correct_transitions_mean = np.array(filtered_correct_transitions_mean)
            filtered_correct_no_changes_mean = np.array(filtered_correct_no_changes_mean)

            # sum_correct_transition_means = 0
            # count_correct_transition_means = 0
            # for mean in correct_transitions_mean:
            #     if mean != "None":
            #         sum_correct_transition_means += mean
            #         count_correct_transition_means += 1
            # sum_correct_no_change_means = 0
            # count_correct_no_change_means = 0
            # for mean in correct_no_changes_mean:
            #     if mean != "None":
            #         sum_correct_no_change_means += mean
            #         count_correct_no_change_means += 1
            overall_transition_mean = round(filtered_correct_transitions_mean.mean(), 2) #sum_correct_transition_means/count_correct_transition_means
            overall_no_change_mean = round(filtered_correct_no_changes_mean.mean(), 2) #sum_correct_no_change_means/count_correct_no_change_means
            overall_transition_std = round(filtered_correct_transitions_mean.std(), 2)
            overall_no_change_std = round(filtered_correct_no_changes_mean.std(), 2)

            print("Transition Accuracy:")
            print("Overall mean accuracy on transitions: {}% (std: {}%)".format(overall_transition_mean, overall_transition_std))
            for i in range(NUM_LIGHTS):
                print("    Light {}: {} ({}/{})".format(i+1, correct_transitions_mean[i], correct_transitions[i], transition_counts[i]))
            print("Overall mean accuracy on no-changes: {}% (std: {}%)".format(overall_no_change_mean, overall_no_change_std))
            for i in range(NUM_LIGHTS):
                print("    Light {}: {} ({}/{})".format(i+1, correct_no_changes_mean[i], correct_no_change[i], no_change_counts[i]))


    if calculate_real_world == True:
        for t_filename in real_world_files:
            print("\nReal-world Accuracy:")
            print("calculating per light accuracy on {}...".format(t_filename))
            x_val = HDF5Matrix(t_filename, input_group)
            y_val = HDF5Matrix(t_filename, output_group)
            num_correct = []
            for i in range(NUM_LIGHTS):
                num_correct.append(0)
            num_tests = len(x_val)
            update_chunk = 20000
            for i in range(num_tests):
                correctness_vector = get_light_accuracies(y_val[i], model.predict(np.reshape(x_val[i], (-1, timesteps, data_dim))))
                for light_index in range(NUM_LIGHTS):
                    num_correct[light_index] += correctness_vector[light_index] # 1 if correct, 0 if not
                if i >= update_chunk and i % update_chunk == 0:
                    print("{} processed of {} ({}%)".format(i, num_tests, round(i/num_tests*100, 2)))
            per_light_percent = []
            for i in range(NUM_LIGHTS):
                per_light_percent.append(round(num_correct[i]/num_tests*100, 2))
            per_light_percent = np.array(per_light_percent)
            mean_percent = round(per_light_percent.mean(), 2)
            std_dev_percent = round(per_light_percent.std(), 2)
            print("Mean percent correct: {}% (std dev: {})".format(mean_percent, std_dev_percent))
            for i in range(NUM_LIGHTS):
                print("\tLight {}: {}% correct".format(i+1, per_light_percent[i]))


# returns those test samples where a light changes in the next state
def get_transitional_samples(x_val, y_val):
    transitional_x = []
    transitional_y = []
    transition_masks = []
    num_samples = len(y_val)
    update_chunk = 20000
    for sample_num in range(1,num_samples):
        s1_lights = y_val[sample_num-1]
        s2_lights = y_val[sample_num]
        transition_mask = []
        light_change = False
        for light_num in range(len(s1_lights)):
            if s1_lights[light_num] != s2_lights[light_num]:
                light_change = True
                transition_mask.append(1)
            else:
                transition_mask.append(0)
        if light_change:
            transitional_x.append(x_val[sample_num])
            transitional_y.append(y_val[sample_num])
            transition_masks.append(transition_mask)
        if sample_num >= update_chunk and sample_num % update_chunk == 0:
            print("{} processed of {} ({}%)".format(sample_num, num_samples, round(sample_num/num_samples*100, 2)))
    return transitional_x, transitional_y, transition_masks

# light accuracies is a list of tuples. Each index is for the light, the tuple there is the mean, std_dev.
def get_light_accuracies(y_true, y_pred):
    y_pred = y_pred[0]
    correctness_vector = []
    for i in range(NUM_LIGHTS):
        correctness_vector.append(0)
    for light_num in range(NUM_LIGHTS):
        if y_true[light_num] == 1 and y_pred[light_num] >= 0.5: # sigmoid
        #if y_true[light_num] == 1 and y_pred[light_num] > 0: # relu
            correctness_vector[light_num] = 1
        elif y_true[light_num] == 0 and y_pred[light_num] < 0.5: #sigmoid
        #elif y_true[light_num] == 0 and y_pred[light_num] == 0: # relu
            correctness_vector[light_num] = 1
    return correctness_vector

main()
