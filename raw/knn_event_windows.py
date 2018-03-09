#this is trying to predict a single light

# Heavily borrowed from https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/

import numpy as np
from numpy import linalg as LA
from collections import Counter
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

    data_dim = len(x_train[0][0])
    num_classes = len(y_train[0])
    num_samples = len(x_train)
    timesteps = len(x_train[0]) #number of timesteps per batch

    print("building model...")

    # TRAIN KNN HERE
    """
    print(type(x_train))
    print("First element of y_train: ", y_train[0][0], "First row of second element of y_train: ", y_train[1][0], " First Element of y_train: ", y_train[0:2])
    print("Distance of first row of x_train1 and x_train2): ", row_vector_distance(y_train[0], y_train[1]))
    test1 = np.zeros((4, 4))
    test2 = np.ones((4, 4))
    print(len(test1.shape))
    print("distance between np array of 0s and 1s: ", row_vector_distance(test1, test2))
    print("matrix distance between np array of 0s and 1s: ", matrices_distance(test1, test2))
    
    """
    #print("First element of x_val: ", x_val[0:3], " First Element of y_val: ", y_val[0:3])
    #print("Length of first element of x_train: ", matrix_distance(x_train[0]))
    #print("Length of third element of x_train: ", matrix_distance(x_train[2]))
    prediction = predict(x_train, y_train, x_val[0], 3)
    print(prediction)
    predictions = kNearestNeighbor(x_train, y_train, x_val[0:3], 3)
    print("First three actual predictions: ", predictions )
    print("First three expected predictions: ", y_val[0:3])
    """

    """
    # Using sklearn.neighbors KNN algorithm, doesn't work bc dimensions off
    #instantiate Learning Model using sklearn.neighbors
    knn = KNeighborsClassifier(n_neighbors = 5)

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
def predict(X_train, y_train, x_test, k):
    distances = []
    targets = []

    for i in range(len(X_train)):
        # compute l2 norm distance between matrices
        # TODO: Want to update this so over all predicts, we have already calculated the distance of each matrix from a "0" matrix and then we only need to subtract the difference between each matrix here
        dist = matrices_distance(X_train[i,:], x_test)
        distances.append([dist, i])

    # sort the list of distances
    distances = sorted(distances)

    # make a list of the k neighbors' targets
    for i in range(k):
        index = distances[i][1]
        targets.append(tuple(y_train[index,:].tolist()))

    # return most common target
    return list(Counter(targets).most_common(1)[0][0])

# Calculates the Euclidean distance btween two 2d numpy arrays
def matrices_distance(m1, m2):
    # if the two matrices are not the same shape, cannot compute the difference in size
    if m1.shape != m2.shape:
        print("ERROR: Cannot calculate distance bewteen matrices of different sizes")
    else:
        # if the matrices are 1d np arrays, calculate the Euclidean difference of the row
        if len(m1.shape) == 1:
            return row_vector_distance(m1, m2)

        # calculate the sum of the euclidean distances of each row
        else:
            vector_distances = []
            for i in range(m1.shape[0]):
                print(i)
                row_dist = row_vector_distance(m1[i,:], m2[i,:])
                vector_distances.append(row_dist)
            return sum(vector_distances)

# Calculates the Euclidean distance between two 1d numpy arrays
def row_vector_distance(v1, v2):
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
