import numpy as np
from scipy.ndimage import zoom, interpolation, rotate, gaussian_filter
import os
import random

# Data preprocessing -----------------------------------------------------------------------
# input (list of ndarray) and return 2d ndarray ( DATA_SIZE * TOTAL_FEATURES )
def flattenImage(train_data_array):
    flatten_list = list(map(lambda x: x.flatten(), train_data_array))
    flatten_array = np.asarray(flatten_list)
    return flatten_array

# input (ndarray) and return 2d ndarray ( DATA_SIZE * TOTAL_FEATURES )
def meanNormalize(train_data_array, batch_size):
    mean = np.sum(train_data_array, 0) / batch_size
    mean_normalized_list = list(map(lambda x: (x - mean) / 255, train_data_array))
    mean_normalized_array = np.asarray(mean_normalized_list)
    return mean_normalized_array

# set constant for dividing training batch
def setBatch(batch_size, total_train_batch):
    batch_list = []
    for i in range(total_train_batch):
        batch_list.append(batch_size*(i+1))
    return (batch_size, batch_list, total_train_batch)


# Data Augmentation -------------------------------------------------------------------------
# return image itself (no augmentation)
def augmentation_identity(image):
    return image

# input unflattened array and return image flipped along y axis
def augmentation_flip(image):
    return np.fliplr(image)

# input unflattened array and return image randomly translated by random distance in range
def augmentation_translate(image):
    return interpolation.shift(image, (random.randint(1,20), random.randint(1,20), 0), order=5, mode='nearest')

# input unflattened array and return image randomly rotated by random degree in range
def augmentation_rotate(image):
    return rotate(image, random.uniform(-30,30), reshape=False, order=5, mode='nearest')

# input unflattened array and return image blurred in several extent
def augmentation_blur(image):
    return gaussian_filter(image, random.randint(1,5))

# Training Utilities ------------------------------------------------------------------------

# input a label and total # of class and return the onehot vector ( 1, totalClass )
def toOnehot(label, totalClass):
    temp = np.zeros((totalClass), dtype=float)
    temp[label] = 1
    return temp

# input a ndarray and perform the element-wise sigmoid
def sigmoid(z, gradient=False):
    if (gradient==True):
        return np.multiply( sigmoid(z), (-sigmoid(z) + 1) )
    else:
        return ( 1 / ( 1 + np.exp(-z) ) )

# input a predicted_result (not one-hot), a correct one-hot result and total # of data and return the cost
def cost(y_predict, y, total_train_data):
    yIsOne = np.multiply((-y), np.log(y_predict))
    yIsZero = np.multiply((1-y), np.log(1 - y_predict))
    return (np.sum(yIsOne-yIsZero) / (-1 * total_train_data))

# input a ndarray without bias column and return 2D ndarray with bias column(row = train data, columns = features + 1(with bias column))
def withBiasColumn(m):
    (row, column) = m.shape
    temp = np.ones((row, column+1))
    temp[:,1:] = m
    return temp

# input a ndarray with bias column and return 2D ndarray without bias column (row = train data, columns = features(without bias column))
def withoutBiasColumn(m):
    (row, column) = m.shape
    temp = np.zeros((row, column-1))
    temp = m[:,1:]
    return temp

# return 1D ndarray (columns = onehot predicted result)
def nn_output_to_onehot_row(output):
    column_index = np.where(output == np.amax(output))[0]
    temp = np.zeros(output.shape)
    temp[column_index] = 1
    return temp

# return 2D ndarray (row = train data, columns = onehot predicted result)
def nn_output_to_onehot(y_predict):
    return np.apply_along_axis(nn_output_to_onehot_row, 1, y_predict)

# return a scalar (the readable label)
def onehot_row_to_label(y_predict):
    return np.where(y_predict == 1)[0][0]

# return 2D ndarray (one column vector) (row = train data, column = readable label)
def onehot_to_label(y_predict):
    return np.apply_along_axis(onehot_row_to_label, 1, y_predict)

# Return 2D array of onehot prediction result
def prediction(X, y, theta, interval=(1, 1)):
    if (not isInterval(interval)):
        return

    input_layer = withBiasColumn(X) # (TOTAL_DATA, 30001)
    hidden_layer_2 = sigmoid(np.dot(input_layer,theta[0].T)) # (TOTAL_DATA, 3000)
    hidden_layer_2 = withBiasColumn(hidden_layer_2) # (TOTAL_DATA, 3001)
    hidden_layer_3 = sigmoid(np.dot(hidden_layer_2,theta[1].T)) # (TOTAL_DATA, 300)
    hidden_layer_3 = withBiasColumn(hidden_layer_3) # (TOTAL_DATA, 301)
    output_layer = sigmoid(np.dot(hidden_layer_3,theta[2].T)) # (TOTAL_DATA, 25)

    return output_layer # (TOTAL_DATA, 25)

# Return accuracy of nn result
def get_accuracy(output, y, TOTAL_DATA, interval=(1, 1)):
    if (not isInterval(interval)):
        return

    prediction = onehot_to_label(nn_output_to_onehot(output))
    return np.sum(prediction == y) / TOTAL_DATA

# Debug Utilities ------------------------------------------------------------------------

# Return boolean
def isInterval(interval):
    (loop, num) = interval
    return (loop % num == 0)

# Print debug message
def debug(str, interval=(1, 1)):
    if (not isInterval(interval)):
        return
    print(str)

# File Output Utilities ------------------------------------------------------------------------

# save total epoch to file
def writeInfo(filepath, interval=(1, 1)):
    if (not isInterval(interval)):
        return

    (loop, num) = interval
    loop = loop + 1

    exists = os.path.isfile(filepath)
    if exists:
        file = open(filepath, "r")
        epoch = int(file.readlines(1)[0].rstrip().split(":")[1])
        file.close()

        os.remove(filepath)

        file = open(filepath, "w")
        file.write("Epoch Trained: " + str(epoch + num) + "\n")
        file.close()

    else:
        file = open(filepath, "w")
        file.write("Epoch Trained: " + str(loop) + "\n")
        file.close()

# save theta to file
def saveTheta(theta, filepath, interval=(1, 1)):
    if (not isInterval(interval)):
        return

    exists = os.path.isfile(filepath)
    if exists:
        os.remove(filepath)

    Theta_temp = np.asarray(theta)
    np.save(filepath, Theta_temp)
