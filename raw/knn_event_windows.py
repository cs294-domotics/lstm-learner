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

    print(type(x_train))
    #print("First element of x_train: ", x_train[0:3], " First Element of y_train: ", y_train[0:3])
    #print("First element of x_val: ", x_val[0:3], " First Element of y_val: ", y_val[0:3])
    #print("Length of first element of x_train: ", matrix_distance(x_train[0]))
    #print("Length of third element of x_train: ", matrix_distance(x_train[2]))
    prediction = predict(x_train, y_train, x_val[0], 3)
    print(prediction)
    predictions = kNearestNeighbor(x_train, y_train, x_val[0:3], 3)
    print(predictions)
    print("First three expected predictions: ", y_val[0:3])

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
    
main()
