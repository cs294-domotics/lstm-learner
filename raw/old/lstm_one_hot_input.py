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
    print(labels[0:10])
    light_id = 4 #first light? second light? etc.
    labels = labels[:,(light_id-1)*2:light_id*2]
    print(labels[0:10])
    sub_problem = np.zeros((len(labels), 3))
    sub_problem[:,:-1] = labels
    for event in sub_problem:
        if np.sum(event) == 0.0:
            event[-1] = 1.0
    labels = sub_problem
    print(labels[0:10])
    turn_on_percent = round((np.sum(labels[:,1])/len(labels[:,1]))*100, 2)
    turn_off_percent = round((np.sum(labels[:,0])/len(labels[:,0]))*100, 2)
    nop_percent = 100 - (turn_off_percent + turn_on_percent)
    labels = np.reshape(labels, (len(labels), 3))
    print("   light turns on " + str(turn_on_percent) + " percent of the time")
    print("   light turns off " + str(turn_off_percent) + " percent of the time")
    print("   accuracy must be higher than " + str(round(nop_percent/100.0, 4)))

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

    num_hidden_states = data_dim
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(num_hidden_states, return_sequences=True, input_shape=(timesteps, data_dim)))
    #model.add(LSTM(num_hidden_states, return_sequences=True))
    model.add(Dense(num_outputs, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=0.01),
                  metrics=['accuracy'])

    print("training and testing model...")
    model.fit(x_train, y_train,
              batch_size=8, epochs=8,
              validation_data=(x_val, y_val))

main()
