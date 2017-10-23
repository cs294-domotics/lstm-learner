from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 4
timesteps = 1
num_classes = 2

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate training data
# dimensions are batch_size, timesteps, data_sim
x_train = np.array([[[0, 0, 0, 1]], # 1
                   [[0, 0, 1, 0]],  # 2
                   [[1, 0, 0, 0]],  # 8
                   [[1, 0, 1, 0]],  # 10
                   [[0, 1, 0, 0]],  # 4
                   [[0, 1, 0, 1]],  # 5
                   [[1, 1, 0, 1]],  # 13
                   [[0, 0, 1, 1]]]) # 3
# two classes:
# [0,1] => "odd", [1,0] => "even"
y_train = np.array([[[0, 1]],       # odd
                    [[1, 0]],       # even
                    [[1, 0]],       # even
                    [[1, 0]],       # even
                    [[1, 0]],       # even
                    [[0, 1]],       # odd
                    [[0, 1]],       # odd
                    [[0, 1]]])      # odd

# Generate validation data
# dimensions are batch_size, timesteps, data_sim
x_val = np.array([[[0, 0, 0, 0]],
                 [[1, 1, 1, 1]],
                 [[1, 1, 0, 0]],
                 [[0, 1, 1, 1]],
                 [[1, 1, 1, 0]],
                 [[0, 1, 1, 0]],
                 [[1, 0, 0, 1]],
                 [[1, 0, 1, 1]]])
y_val = np.array([[[1,0]],
                  [[0,1]],
                  [[1,0]],
                  [[0,1]],
                  [[1,0]],
                  [[1,0]],
                  [[0,1]],
                  [[0,1]]])

model.fit(x_train, y_train,
          batch_size=8, epochs=8,
          validation_data=(x_val, y_val))
