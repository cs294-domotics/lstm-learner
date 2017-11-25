from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop

#import seq2seq
#from seq2seq.models import SimpleSeq2Seq

import tensorflow as tf

import numpy as np

folder = "build/"
train_input_filename = folder + "twor2010_1s_one_week_train_in.npy"
train_output_filename = folder + "twor2010_1s_one_week_train_out.npy"
test_input_filename = folder + "twor2010_1s_one_week_test_in.npy"
test_output_filename = folder + "twor2010_1s_one_week_test_out.npy"


def main():

    print("loading data...")
    # Generate training data
    # dimensions are batch_size, timesteps, data_sim
    x_train = np.load(train_input_filename)
    y_train = np.load(train_output_filename)

    # Generate validation data
    # dimensions are batch_size, timesteps, data_sim
    x_val = np.load(test_input_filename)
    y_val = np.load(test_output_filename)

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

    num_hidden_states = 256
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(data_dim, return_sequences=True, input_shape=(timesteps, data_dim)))
    #model.add(Dense(num_outputs, activation='tanh'))

    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop(lr=0.01),
                  metrics=['mae', 'categorical_accuracy'])

    print(model.summary())


    #model = SimpleSeq2Seq(input_length=timesteps, input_dim=data_dim, hidden_dim=20, output_length=timesteps, output_dim=data_dim)
    #model.compile(loss='mse', optimizer='rmsprop')

    print("training and testing model...")
    model.fit(x_train, y_train,
              batch_size=8, epochs=8,
              validation_data=(x_val, y_val))

    # print("calculating custom accuracy...")
    # acc = 0.0
    # num_val = len(x_val)
    # for i in range(num_val):
    #     acc += light_accuracy(y_val[i], model.predict(np.reshape(x_val[i], (-1, timesteps, data_dim))))
    #     #print(model.predict(np.reshape(x_val[0], (-1, timesteps, data_dim))))
    #     #print(y_val[0])
    # acc = acc/num_val
    # print("mean light accuracy: " + str(acc))


def light_accuracy(y_true, y_pred):
    num_correct = 0.0
    num_lights = 11
    pred_light_states = y_pred[0][-1][-11:] # last 11 entries on the last row
    true_light_states = y_true[-1][-11:]
    for i in range(num_lights):
        if pred_light_states[i] >= 0.5 and true_light_states[i] == 1:
            num_correct += 1
        elif pred_light_states[i] < 0.5 and true_light_states[i] == 0:
            num_correct += 1
    return num_correct/num_lights

main()
