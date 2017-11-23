#this is trying to predict a single light

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop
import numpy as np

features_filename = "features.npy"
labels_filename = "labels.npy"


def main():
    print("loading data...")
    features = np.load("features.npy")
    labels = np.load("labels.npy")
    # select one light to learn
    labels = labels[:,0]
    on_percent = round((np.sum(labels)/len(labels))*100, 2)
    off_percent = 100 - on_percent
    labels = np.reshape(labels, (len(labels), 1))
    print("   light is on " + str(on_percent) + " percent of the time")
    print("   light is off " + str(off_percent) + " percent of the time")

    # build the actual class labels (e.g., for each timestep one-hot [is_off, is_on])
    #class_labels = np.zeros((len(labels), 2))
    #for (t, label) in enumerate(labels):
    #    class_labels[t][int(label)] = 1 #this assumes the label is either 0 for on or 1 for off
    #labels = class_labels

    print("data loaded...")

    if len(features) != len(labels):
        print("uh oh...features and labels are different sizes.")

    print("generating training data...")

    data_dim = len(features[0])
    num_outputs = len(labels[0])
    num_samples = len(features)
    timesteps = 1 #number of timesteps per batch


    # Generate training data
    # dimensions are batch_size, timesteps, data_sim
    x_train = features[:int(num_samples/2)]
    x_train = x_train[:, None, :]

    # two classes:
    # [0,1] => "odd", [1,0] => "even"
    y_train = labels[:int(num_samples/2)]
    y_train = y_train[:, None, :]

    print("generating test data...")
    # Generate validation data
    # dimensions are batch_size, timesteps, data_sim
    x_val = features[int(num_samples/2):]
    x_val = x_val[:, None, :]
    y_val = labels[int(num_samples/2):]
    y_val = y_val[:, None, :]

    print("building model...")

    num_hidden_states = 32
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(num_hidden_states, return_sequences=True, input_shape=(timesteps, data_dim)))
    model.add(LSTM(num_hidden_states, return_sequences=True))
    #model.add(Dense(num_outputs, activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.01),
                  metrics=['accuracy'])

    print("training and testing model...")
    model.fit(x_train, y_train,
              batch_size=8, epochs=8,
              validation_data=(x_val, y_val))

main()
