#this is trying to predict a single light

# Heavily borrowed from https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/

"""
TODO: (1) check if euclidean distance is accurate on trained values using k = 1 and an accuracy function
      (2) incorporate numpy arrays better
      (3) try to optimize code
      (4) maybe used weighted knn bc right now the k nearest neighbors have equal weight, but what happens if some further neighbors skew value to inaccurate value
      (5) using most_common in the collections counter library chooses arbitrarily in case of ties (i.e. 2 things appear the same amount of times, most_common arbitrarily chooses one)
"""
import numpy as np
from numpy import linalg as LA
from collections import Counter
from time import time
#from sklearn.neighbors import KNeighborsClassifier

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
load_folder = "build/events/raw/light_and_time/"

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

# row size of an array in the test y_test/val matrix
y_test_row_size = 3

def main():

    print("loading data...")

    # Training Data
    x_train = np.load(input_train_filename)
    y_train = np.load(output_train_filename)

    # Testing Data
    x_val = np.load(input_test_filename)
    y_val = np.load(output_test_filename)

    print("data loaded...")

    print("x_train's shape: ", x_train.shape)
    print("x_val's shape: ", x_val.shape)
    print("y_train's shape: ", y_train.shape)
    print("y_val's shape: ", y_val.shape)

    if len(x_train) != len(y_train):
        print("uh oh...features and labels are different sizes.")

    data_dim = len(x_train[0][0])
    num_classes = len(y_train[0])
    num_samples = len(x_train)
    timesteps = len(x_train[0]) #number of timesteps per batch

    print("building model...")

    # TRAIN KNN HERE
    
    """
    #print(type(x_train))
    #print("First element of y_train: ", y_train[0][0], "First row of second element of y_train: ", y_train[1][0], " First Element of y_train: ", y_train[0:2])
    #print("Distance of first row of x_train1 and x_train2): ", row_vector_distance(y_train[0], y_train[1]))
    
    #TESTS TO SEE IF matrices_distance and row_vector_distance WORK
    test1 = np.zeros((4, 4))
    test2 = np.ones((4, 4))
    test2 = np.add(test2, test2)
    print((test1.shape), " ", len(test1.shape))
    print("distance between np array of 0s and 1s: ", row_vector_distance(test1, test2))
    print("matrix distance between np array of 0s and 1s: ", matrices_distance(test1, test2))
    
    """
    start = 0
    """
    print("======================Predict function testing========================")
    #print(start, " index element of x_val: ", x_train[start], start, " index element of y_val: ", y_train[start])
    starttime = time()
    prediction = predict(x_train, y_train, x_val[start], 3)
    print(prediction)
    print(y_val[start])
    print(time() - starttime)
    """
    #Tests for kNearestNeighbor
    print("==============Test for kNearestNeighbor=================")
    #print("X_train: ", x_train)
    starttime = time()
    predictions = kNearestNeighbor(x_train, y_train, x_val[start: start + 3], 3)
    print("First three actual predictions: ", predictions)
    print("First three expected predictions: ", y_val[start: start + 3])
    predictions = kNearestNeighbor(x_train, y_train, x_val, 3)
    print(time() - starttime)
    print("First three actual predictions: ", predictions)
    """
    #print("First three expected predictions: ", y_val)
    # fitting the model
    knn.fit(x_train, y_train)


    print("training and testing model...")

    # TEST KNN HERE

    pred = knn.predict(x_val)

    print("Accuracy score: ", accuracy_score(y_val, pred))
    """

def kNearestNeighbor(X_train, y_train, X_test, k):
    # train on input data

    # loop over all observations
    predictions = []
    for i in range(len(X_test)):
        predictions.append(predict(X_train, y_train, X_test[i,:], k))
    
    # return the predictions made
    return predictions


# Function that predicts where an unsigned label will go based on training data
""" Parameters: X_train: entire x_train matrix that we train over (3d array)
y_train: entire y_train matrix we train over (2d array)
x_test: single x_matrix we test (2d array)
k: number of nearest neighbors we use to decide what prediction x_test will be (MUST BE ODD)
"""
def predict(X_train, y_train, x_test, k):
    distances = []
    targets = []

    for i in range(len(X_train)):
        # compute l2 norm distance between matrices
        # TODO: Want to update this so over all predicts, we have already calculated the distance of each matrix from a "0" matrix and then we only need to subtract the difference between each matrix here
        #print(X_train)
        #print("X_train shape: ", X_train[i,:].shape, "xtest shape: ", x_test.shape)
        dist = matrices_distance(X_train[i,:], x_test)
        distances.append([dist, i])
    #print(distances)

    # sort the list of distances based on first element of each in ASC order
    distances = sorted(distances)

    #print(y_train)
    #print (distances[:10])
    # make a list of indices of the k neighbors' targets
    for i in range(k):
        index = distances[i][1]
        targets.append(tuple(y_train[index,:].tolist()))
    
    print(targets)
    # return most common target
    # TODO: bug, if more than one thing appears at the same frequency, arbitrarily chooses one
    # Probably want to weight it so the one with the closest distance is picked instead
    return list(Counter(targets).most_common(1)[0][0])

# Calculates the Euclidean distance btween two 2d numpy arrays
def matrices_distance(m1, m2):
    # if the two matrices are not the same shape, cannot compute the difference in size
    if m1.shape != m2.shape:
        print("ERROR: Cannot calculate distance bewteen matrices of different sizes. M1 shape: ", m1.shape, ". M2 shape: ", m2.shape)
        return
    # if the matrices are 1d np arrays, calculate the Euclidean difference of the row
    if len(m1.shape) == 1:
        return row_vector_distance(m1, m2)

    # calculate the sum of the euclidean distances of each row
    else:
        vector_distances = []
        for i in range(m1.shape[0]):
            row_dist = row_vector_distance(m1[i,:], m2[i,:])
            vector_distances.append(row_dist)
        return sum(vector_distances)

def matrix_distance(m1):
    return matrices_distance(m1, np.zeros(m1.shape)) 

# Calculates the Euclidean distance between two 1d numpy arrays
def row_vector_distance(v1, v2):
    #print(v1)
    #print(v2)
    if v1.shape != v2.shape:
        print("ERROR: cannot calculate distance between vectors of different sizes")
        return
    if len(v1.shape) != 1:
        print("ERROR: thing passed in is not a vector")
        return
    return np.sqrt(np.sum((v1 - v2) ** 2))

"""
# Old Distance metric, just calculates size of each individual row and finds the difference in size between two matrices. ** Not super accurate 

# Calculates distance between two matrices based on the difference in each row
def matrices_distance(m1, m2):
    m1_distances = matrix_distance(m1)
    m2_distances = matrix_distance(m2)
    m1_distance = sum(m1_distances)
    m2_distance = sum(m2_distances)
    return m1_distance - m2_distance

# Returns a list of the lengths of each row of one matrix
def matrix_distance(matrix):
    v_distances = []
    for i in range(len(matrix)):
        row_dist = LA.norm(matrix[i,:])   # Calculates the Euclidean/Frobenius norm
        v_distances.append(row_dist)
    return v_distances
"""

main()
