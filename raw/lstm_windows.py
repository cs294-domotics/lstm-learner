#this is trying to predict a single light

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop
import numpy as np

features_filename = "build/L005_features.npy"
labels_filename = "build/L005_labels.npy"


def main():
    print("loading data...")
    features = np.load(features_filename)
    labels = np.load(labels_filename)

    turn_on_percent = round((np.sum(labels[:,1])/len(labels[:,1]))*100, 2)
    turn_off_percent = round((np.sum(labels[:,0])/len(labels[:,0]))*100, 2)
    nop_percent = round((np.sum(labels[:,2])/len(labels[:,2]))*100, 2)
    print("   light turns on " + str(turn_on_percent) + " percent of the time")
    print("   light turns off " + str(turn_off_percent) + " percent of the time")
    print("   light does nothing " + str(turn_off_percent) + " percent of the time")
    print("   accuracy must be higher than " + str(round(max(turn_on_percent, turn_off_percent, nop_percent)/100.0, 4)))

    print("data loaded...")

    if len(features) != len(labels):
        print("uh oh...features and labels are different sizes.")

    print("generating training data...")

    data_dim = len(features[0][0])
    num_classes = len(labels[0])
    num_samples = len(features)
    timesteps = len(features[0]) #number of timesteps per batch

    #labels = labels.reshape((num_samples, None, num_classes))


    # Generate training data
    # dimensions are batch_size, timesteps, data_sim
    x_train = features[:int(num_samples/2)]
    y_train = labels[:int(num_samples/2)]

    print("generating test data...")
    # Generate validation data
    # dimensions are batch_size, timesteps, data_sim
    x_val = features[int(num_samples/2):]
    y_val = labels[int(num_samples/2):]

    print("building model...")

    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    num_hidden_states = data_dim
    model.add(LSTM(num_hidden_states, return_sequences=False, input_shape=(timesteps, data_dim)))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=0.01),
                  metrics=['accuracy'])

    print("training and testing model...")
    model.fit(x_train, y_train,
              batch_size=timesteps, epochs=50,
              validation_data=(x_val, y_val))

main()
