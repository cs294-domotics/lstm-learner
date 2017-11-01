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
    print("data loaded...")

    if len(features) != len(labels):
        print("uh oh...features and labels are different sizes.")

    print("generating training data...")
    data_dim = len(features[0])
    num_outputs = len(labels[0]) #numlights
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

    num_hidden_states = 20
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(num_hidden_states, return_sequences=True, input_shape=(timesteps, data_dim)))
    #model.add(LSTM(num_outputs, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(Dense(num_outputs, activation='linear'))

    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop(lr=0.01),
                  metrics=['accuracy'])

    print("training and testing model...")
    model.fit(x_train, y_train,
              batch_size=8, epochs=8,
              validation_data=(x_val, y_val))

main()
