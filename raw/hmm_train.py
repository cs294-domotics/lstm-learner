#this is trying to predict a single light

import numpy as np

#features_filename = "build/events/raw/no_light_no_time/L005_20_features.npy"
#labels_filename = "build/events/raw/no_light_no_time/L005_20_labels.npy"
#features_filename = "build/events/raw/light_and_time/L005_20_features.npy"
#labels_filename = "build/events/raw/light_and_time/L005_20_labels.npy"
#features_filename = "build/events/activities/no_light_no_time/L005_20_features.npy"
#labels_filename = "build/events/activities/no_light_no_time/L005_20_labels.npy"
#features_filename = "build/events/activities/light_and_time/L005_20_features.npy"
#labels_filename = "build/events/activities/light_and_time/L005_20_labels.npy"
#features_filename = "build/events/activities/light_and_time/L005_40_features.npy"
#labels_filename = "build/events/activities/light_and_time/L005_40_labels.npy"

#load_folder = "build/events/activities/light_and_time/"
load_folder = "build/events/raw/no_light_no_time/"

##### TRAIN

#input_train_filename = load_folder + "L005_40_one_day_train_features.npy"
#output_train_filename = load_folder + "L005_40_one_day_train_labels.npy"

#input_train_filename = load_folder + "L005_40_one_week_train_features.npy"
#output_train_filename = load_folder + "L005_40_one_week_train_labels.npy"

#input_train_filename = load_folder + "l005_40_two_weeks_train_features.npy"
#output_train_filename = load_folder + "l005_40_two_weeks_train_labels.npy"

#input_train_filename = load_folder + "L005_40_one_month_train_features.npy"
#output_train_filename = load_folder + "L005_40_one_month_train_labels.npy"

#input_train_filename = load_folder + "L005_5_one_day_train_features.npy"
#output_train_filename = load_folder + "L005_5_one_day_train_labels.npy"

input_train_filename = load_folder + "L005_5_one_month_train_features.npy"
output_train_filename = load_folder + "L005_5_one_month_train_labels.npy"


##### TESTS

#input_test_filename = load_folder + "L005_40_one_day_test_features.npy"
#output_test_filename = load_folder + "L005_40_one_day_test_labels.npy"

#input_test_filename = load_folder + "L005_40_one_week_test_features.npy"
#output_test_filename = load_folder + "L005_40_one_week_test_labels.npy"

#input_test_filename = load_folder + "L005_40_two_weeks_test_features.npy"
#output_test_filename = load_folder + "L005_40_two_weeks_test_labels.npy"

#input_test_filename = load_folder + "L005_40_one_month_test_features.npy"
#output_test_filename = load_folder + "L005_40_one_month_test_labels.npy"

#input_test_filename = load_folder + "L005_5_one_day_test_features.npy"
#output_test_filename = load_folder + "L005_5_one_day_test_labels.npy"

#input_test_filename = load_folder + "L005_5_two_weeks_test_features.npy"
#output_test_filename = load_folder + "L005_5_two_weeks_test_labels.npy"

input_test_filename = load_folder + "L005_5_one_month_test_features.npy"
output_test_filename = load_folder + "L005_5_one_month_test_labels.npy"


def main():

    print("loading data...")

    x_train = np.load(input_train_filename)
    y_train = np.load(output_train_filename)

    x_val = np.load(input_test_filename)
    y_val = np.load(output_test_filename)

    print("data loaded...")

    print(x_train.shape)
    print(x_val.shape)
    print(y_train.shape)
    print(y_val.shape)

    if len(x_train) != len(y_train):
        print("uh oh...features and labels are different sizes.")


main()
