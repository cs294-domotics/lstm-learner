#this is trying to predict a single light

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop
from keras import regularizers
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

load_folder = "build/events/activities/light_and_time/"

##### TRAIN

#input_train_filename = load_folder + "L005_40_one_day_train_features.npy"
#output_train_filename = load_folder + "L005_40_one_day_train_labels.npy"

#input_train_filename = load_folder + "L005_40_one_week_train_features.npy"
#output_train_filename = load_folder + "L005_40_one_week_train_labels.npy"

#input_train_filename = load_folder + "L005_40_two_weeks_train_features.npy"
#output_train_filename = load_folder + "L005_40_two_weeks_train_labels.npy"

input_train_filename = load_folder + "L005_40_one_month_train_features.npy"
output_train_filename = load_folder + "L005_40_one_month_train_labels.npy"

##### TESTS

#input_test_filename = load_folder + "L005_40_one_day_test_features.npy"
#output_test_filename = load_folder + "L005_40_one_day_test_labels.npy"

#input_test_filename = load_folder + "L005_40_one_week_test_features.npy"
#output_test_filename = load_folder + "L005_40_one_week_test_labels.npy"

#input_test_filename = load_folder + "L005_40_two_weeks_test_features.npy"
#output_test_filename = load_folder + "L005_40_two_weeks_test_labels.npy"

input_test_filename = load_folder + "L005_40_one_month_test_features.npy"
output_test_filename = load_folder + "L005_40_one_month_test_labels.npy"


def main():

    print("loading data...")

    x_train = np.load(input_train_filename)
    y_train = np.load(output_train_filename)

    x_val = np.load(input_test_filename)
    y_val = np.load(output_test_filename)

    # turn_on_percent = round((np.sum(labels[:,1])/len(labels[:,1]))*100, 2)
    # turn_off_percent = round((np.sum(labels[:,0])/len(labels[:,0]))*100, 2)
    # nop_percent = round((np.sum(labels[:,2])/len(labels[:,2]))*100, 2)
    # print("   light turns on " + str(turn_on_percent) + " percent of the time")
    # print("   light turns off " + str(turn_off_percent) + " percent of the time")
    # print("   light does nothing " + str(nop_percent) + " percent of the time")
    # print("   accuracy must be higher than " + str(round(max(turn_on_percent, turn_off_percent, nop_percent)/100.0, 4)))
    #
    print("data loaded...")

    print(x_train.shape)
    print(x_val.shape)
    print(y_train.shape)
    print(y_val.shape)

    if len(x_train) != len(y_train):
        print("uh oh...features and labels are different sizes.")

    data_dim = len(x_train[0][0])
    num_classes = len(y_train[0])
    num_samples = len(x_train)
    timesteps = len(x_train[0]) #number of timesteps per batch

    # turn_on_percent = round((np.sum(y_train[:,1])/len(y_train[:,1]))*100, 2)
    # turn_off_percent = round((np.sum(y_train[:,0])/len(y_train[:,0]))*100, 2)
    # nop_percent = round((np.sum(y_train[:,2])/len(y_train[:,2]))*100, 2)
    # print("   light turns on " + str(turn_on_percent) + " percent of the time")
    # print("   light turns off " + str(turn_off_percent) + " percent of the time")
    # print("   light does nothing " + str(nop_percent) + " percent of the time")
    # print("   accuracy must be higher than " + str(round(max(turn_on_percent, turn_off_percent, nop_percent)/100.0, 4)))

    # turn_on_percent = round((np.sum(y_val[:,1])/len(y_val[:,1]))*100, 2)
    # turn_off_percent = round((np.sum(y_val[:,0])/len(y_val[:,0]))*100, 2)
    # nop_percent = round((np.sum(y_val[:,2])/len(y_val[:,2]))*100, 2)
    # print("   light turns on " + str(turn_on_percent) + " percent of the time")
    # print("   light turns off " + str(turn_off_percent) + " percent of the time")
    # print("   light does nothing " + str(nop_percent) + " percent of the time")
    # print("   accuracy must be higher than " + str(round(max(turn_on_percent, turn_off_percent, nop_percent)/100.0, 4)))

    print("building model...")

    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    #num_hidden_states = data_dim
    num_hidden_states = 20
    model.add(LSTM(num_hidden_states, return_sequences=False, input_shape=(timesteps, data_dim)))
    model.add(Dense(num_classes, kernel_regularizer=regularizers.l2(0.01),
    activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=0.01),
                  metrics=['accuracy'])

    print("training and testing model...")
    model.fit(x_train, y_train,
              batch_size=8, epochs=5,
              validation_data=(x_val, y_val))

main()
