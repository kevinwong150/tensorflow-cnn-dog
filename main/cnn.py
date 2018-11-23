import os
import tensorflow as tf
import numpy as np
import util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(BATCH_SIZE, BATCH_LIST, TOTAL_TRAIN_BATCH) = (None, None, None)
LEARNING_RATE = 0.03
TOTAL_CLASS = 25 # Configurable
INPUT_LAYER = 30000 # Configurable
LAYER_2 = 10000 # Configurable
LAYER_3 = 3000 # Configurable
OUTPUT_LAYER = TOTAL_CLASS

TEST_DATA_PATH = "../data/testing.npy"
TRAIN_DATA_PATH = "../data/train_data_array.npy"
TRAIN_LABEL_PATH = "../data/train_label_onehot.npy"
TRAIN_THETA_PATH = "../data/trained_theta.npy"
TRAIN_INFO_PATH = "../data/training_info.txt"

# Load Test Data + Test Label
Dictionary = np.load(TEST_DATA_PATH, encoding="latin1").item()
print("Test set dictionary loaded.")
# (['shapes', 'file_name', 'original', 'reshaped', 'label'])
test_label = (Dictionary['label'])
test_label = np.asarray(test_label)
test_label_onehot = list(map(lambda x: util.toOnehot(x, TOTAL_CLASS), test_label))
test_label_onehot = np.asarray(test_label_onehot) # (250, 25)
(TOTAL_TEST_DATA, _) = test_label_onehot.shape

test_data_array = (Dictionary['reshaped'])
test_data_array = list(map(lambda x: x.astype(np.uint8).flatten(), test_data_array))
test_data_array = np.asarray(test_data_array) # (250, 30000)
test_data_array_mean = np.sum(test_data_array, axis=0)/TOTAL_TEST_DATA
test_data_array = list(map(lambda x: (x - test_data_array_mean) / 255, test_data_array.tolist()))
test_data_array = np.asarray(test_data_array) # (250, 30000)

# Load Train Data + Train Label
train_label_onehot = np.load(TRAIN_LABEL_PATH)
train_label = util.onehot_to_label(train_label_onehot)
train_data_array = np.load(TRAIN_DATA_PATH)

# Get total # of data
(TOTAL_TRAIN_DATA, _) = train_label_onehot.shape

# adjust the batch-size and batch number for training, (maximum batch size = TOTAL_TRAIN_DATA, maximun batch number * batch size <= TOTAL_TRAIN_DATA)
(BATCH_SIZE, BATCH_LIST, TOTAL_TRAIN_BATCH) = util.setBatch(1000, 3) # Configurable


x = tf.placeholder(tf.float32, [None, INPUT_LAYER])
W1 = tf.Variable(tf.zeros([INPUT_LAYER, LAYER_2]))
b1 = tf.Variable(tf.zeros([LAYER_2]))
W2 = tf.Variable(tf.zeros([LAYER_2, LAYER_3]))
b2 = tf.Variable(tf.zeros([LAYER_3]))
W3 = tf.Variable(tf.zeros([LAYER_3, OUTPUT_LAYER]))
b3 = tf.Variable(tf.zeros([OUTPUT_LAYER]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, OUTPUT_LAYER])
cross_entropy = tf.losses.mean_squared_error(y, y_)
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(100):

    util.debug("Epoch #" + str(epoch))

    X = train_data_array
    y = train_label_onehot

    shuffle_order = np.arange(TOTAL_TRAIN_DATA)
    np.random.shuffle(shuffle_order)
    X_shuffle = X[shuffle_order]
    X_batch = np.split(X_shuffle, BATCH_LIST) # batch size = (1000, 1000, 1000, 1728)
    y_shuffle = y[shuffle_order]
    y_batch = np.split(y_shuffle, BATCH_LIST) # batch size = (1000, 1000, 1000, 1728)
    train_label_shuffle = train_label[shuffle_order]
    train_label_batch = np.split(train_label_shuffle, BATCH_LIST) # batch size = (1000, 1000, 1000, 1728)

    for iteration in range(TOTAL_TRAIN_BATCH):
        X_batch[iteration] = util.flattenImage(X_batch[iteration])
        X_batch[iteration] = util.meanNormalize(X_batch[iteration], BATCH_SIZE)

        sess.run(train_step, feed_dict = {x: X_batch[iteration], y_: y_batch[iteration]})
        print(sess.run(accuracy, feed_dict={x: X_batch[iteration], y_: y_batch[iteration]}))
