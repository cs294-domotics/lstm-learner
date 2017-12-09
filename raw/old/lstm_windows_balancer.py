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
# features_filename = "build/events/activities/light_and_time/L005_40_features.npy"
# labels_filename = "build/events/activities/light_and_time/L005_40_labels.npy"
#features_filename = "build/events/activities/light_and_time/L005_80_features.npy"
#labels_filename = "build/events/activities/light_and_time/L005_80_labels.npy"
#features_filename = "build/events/activities/light_and_time/L005_60_features.npy"
#labels_filename = "build/events/activities/light_and_time/L005_60_labels.npy"
#features_filename = "build/events/activities/light_and_time/L005_40_features.npy"
#labels_filename = "build/events/activities/light_and_time/L005_40_labels.npy"
#features_filename = "build/events/raw/no_light_no_time/L005_40_features.npy"
#labels_filename = "build/events/raw/no_light_no_time/L005_40_labels.npy"
features_filename = "build/events/raw/light_and_time/L005_40_features.npy"
labels_filename = "build/events/raw/light_and_time/L005_40_labels.npy"

def main():
    print("loading data...")
    features = np.load(features_filename)
    labels = np.load(labels_filename)

    turn_on_percent = round((np.sum(labels[:,1])/len(labels[:,1]))*100, 2)
    turn_off_percent = round((np.sum(labels[:,0])/len(labels[:,0]))*100, 2)
    nop_percent = round((np.sum(labels[:,2])/len(labels[:,2]))*100, 2)
    print("   light turns on " + str(turn_on_percent) + " percent of the time")
    print("   light turns off " + str(turn_off_percent) + " percent of the time")
    print("   light does nothing " + str(nop_percent) + " percent of the time")
    print("   accuracy must be higher than " + str(round(max(turn_on_percent, turn_off_percent, nop_percent)/100.0, 4)))

    print("data loaded...")

    if len(features) != len(labels):
        print("uh oh...features and labels are different sizes.")

    print("generating training data...")

    data_dim = len(features[0][0])
    num_classes = len(labels[0])
    num_samples = len(features)
    timesteps = len(features[0]) #number of timesteps per batch


    class_features = []
    for i in range(num_classes):
        class_features.append([])
    class_labels = []
    for i in range(num_classes):
        class_labels.append([])

    for i in range(len(features)):
        for j in range(num_classes):
            if labels[i][j] == 1:
                class_labels[j].append(labels[i])
                class_features[j].append(features[i])

    # Generate test data
    # dimensions are batch_size, timesteps, data_sim
    x_train = []
    for class_examples in class_features:
        half = int(len(class_examples)/2)
        x_train += class_examples[:half]
    x_train = np.array(x_train)
    y_train = []
    for class_examples in class_labels:
        half = int(len(class_examples)/2)
        y_train += class_examples[:half]
    y_train = np.array(y_train)
    print(features.shape)
    print(y_train.shape)


    turn_on_percent = round((np.sum(y_train[:,1])/len(y_train[:,1]))*100, 2)
    turn_off_percent = round((np.sum(y_train[:,0])/len(y_train[:,0]))*100, 2)
    nop_percent = round((np.sum(y_train[:,2])/len(y_train[:,2]))*100, 2)
    print("   light turns on " + str(turn_on_percent) + " percent of the time")
    print("   light turns off " + str(turn_off_percent) + " percent of the time")
    print("   light does nothing " + str(nop_percent) + " percent of the time")
    print("   accuracy must be higher than " + str(round(max(turn_on_percent, turn_off_percent, nop_percent)/100.0, 4)))

    print("generating test data...")
    # Generate validation data
    # dimensions are batch_size, timesteps, data_sim
    x_val = []
    for class_examples in class_features:
        half = int(len(class_examples)/2)
        x_val += class_examples[half:]
    x_val = np.array(x_val)
    y_val = []
    for class_examples in class_labels:
        half = int(len(class_examples)/2)
        y_val += class_examples[half:]
    y_val = np.array(y_val)
    print(y_val.shape)

    turn_on_percent = round((np.sum(y_val[:,1])/len(y_val[:,1]))*100, 2)
    turn_off_percent = round((np.sum(y_val[:,0])/len(y_val[:,0]))*100, 2)
    nop_percent = round((np.sum(y_val[:,2])/len(y_val[:,2]))*100, 2)
    print("   light turns on " + str(turn_on_percent) + " percent of the time")
    print("   light turns off " + str(turn_off_percent) + " percent of the time")
    print("   light does nothing " + str(nop_percent) + " percent of the time")
    print("   accuracy must be higher than " + str(round(max(turn_on_percent, turn_off_percent, nop_percent)/100.0, 4)))

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
