import os
import tensorflow as tf
import numpy as np
import util
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
TOTAL_TRAIN_DATA = None
LEARNING_RATE = 0.003
(BATCH_SIZE, BATCH_LIST, TOTAL_TRAIN_BATCH) = (None, None, None)
AUGMENTATION_METHOD = [
    util.augmentation_flip,
    util.augmentation_translate,
    util.augmentation_rotate,
    util.augmentation_blur,
    util.augmentation_identity
    ] # Configurable
TOTAL_AUGMENTATION_METHOD = len(AUGMENTATION_METHOD)
PROGRAM_START_TIME = time.time()
DEBUG_EPOCH_INTERVAL = 5 # Configurable
RANDOM_SEED = 1 # Configurable

TOTAL_CLASS = 25 # Configurable
IMAGE_SHAPE = [None, 100, 100, 3] # Configurable
INPUT_LAYER = 30000 # Configurable
LAYER_2 = 3000 # Configurable
LAYER_3 = 300 # Configurable
OUTPUT_LAYER = TOTAL_CLASS

TEST_DATA_PATH = "../data/testing.npy"
TRAIN_DATA_PATH = "../data/train_data_array.npy"
TRAIN_LABEL_PATH = "../data/train_label_onehot.npy"
TRAIN_MODEL_PATH = "../data/trained_model.ckpt"
TRAIN_INFO_PATH = "../data/training_info.txt"

# Load Test Data + Test Label ------------------------------------------------------------
Dictionary = np.load(TEST_DATA_PATH, encoding="latin1").item()
print("Test set dictionary loaded.")
# (['shapes', 'file_name', 'original', 'reshaped', 'label'])
test_label = (Dictionary['label'])
test_label = np.asarray(test_label)
test_label_onehot = list(map(lambda x: util.toOnehot(x, TOTAL_CLASS), test_label))
test_label_onehot = np.asarray(test_label_onehot) # (250, 25)

test_data_array = (Dictionary['reshaped'])
test_data_array = list(map(lambda x: x.astype(np.uint8).reshape([100, 100, 3]), test_data_array))
test_data_array = np.asarray(test_data_array) # (250, 100, 100, 3)

# Load Train Data + Train Label ------------------------------------------------------------
train_label_onehot = np.load(TRAIN_LABEL_PATH)
train_label = util.onehot_to_label(train_label_onehot)
train_data_array = np.load(TRAIN_DATA_PATH)

# Get total # of data
(TOTAL_TRAIN_DATA, _) = train_label_onehot.shape

# Tensorflow --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Input placeholder ------------------------------------------------------------
image_batch = tf.placeholder("float", IMAGE_SHAPE)
correct_label_onehot = tf.placeholder("float", [None, TOTAL_CLASS])
is_training = tf.placeholder(tf.bool, shape=())

# Preprocess data ------------------------------------------------------------
# # 1. per-image normalize, this will get weirdy good validation result, reasons' unknown
# normalize_image_batch = tf.map_fn(lambda image: tf.image.per_image_standardization(image), image_batch)
# 2. batch normalize
normalize_image_batch = tf.layers.batch_normalization(image_batch, training=is_training)

flatten_image_batch = tf.map_fn(lambda image: tf.reshape(image, [-1]), normalize_image_batch)

# Neural network design ------------------------------------------------------------
# Input layer = 30000 features + 1 bias
# Output Layer = 25 class
# Designed neural network = 4 layer = 30001 + 3001 + 301 + 25
input_layer = flatten_image_batch
input_layer_weight = tf.Variable(tf.random.normal([INPUT_LAYER, LAYER_2]))
input_layer_bias = tf.Variable(tf.random.normal([LAYER_2]))

hidden_layer_2 = tf.add(tf.matmul(input_layer, input_layer_weight), input_layer_bias)
hidden_layer_2_weight = tf.Variable(tf.random.normal([LAYER_2, LAYER_3]))
hidden_layer_2_bias = tf.Variable(tf.random.normal([LAYER_3]))

hidden_layer_3 = tf.add(tf.matmul(hidden_layer_2, hidden_layer_2_weight), hidden_layer_2_bias)
hidden_layer_3_weight = tf.Variable(tf.random.normal([LAYER_3, OUTPUT_LAYER]))
hidden_layer_3_bias = tf.Variable(tf.random.normal([OUTPUT_LAYER]))

output_layer = tf.add(tf.matmul(hidden_layer_3, hidden_layer_3_weight), hidden_layer_3_bias)


# Training Step ------------------------------------------------------------
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=correct_label_onehot))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

# Get Evaluation Metrics ------------------------------------------------------------
# true_positive = tf.equal(tf.argmax(output_layer, 1), tf.argmax(correct_label_onehot, 1))
# accuracy = tf.reduce_mean(tf.cast(true_positive, tf.float32))
output_layer_onehot = tf.one_hot(tf.argmax(output_layer, 1), 25)
accuracy, update_op_accuracy = tf.metrics.accuracy(correct_label_onehot, output_layer_onehot, name="accuracy")
precision, update_op_precision = tf.metrics.precision(correct_label_onehot, output_layer_onehot, name="precision")
recall, update_op_recall = tf.metrics.recall(correct_label_onehot, output_layer_onehot, name="recall")
reset_vars_list = \
    tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy") + \
    tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="precision") + \
    tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="recall")
reset_vars_initializer = tf.initializers.variables(var_list=reset_vars_list)


# Set util variable ------------------------------------------------------------
init = tf.global_variables_initializer()
saver = tf.train.Saver()

(BATCH_SIZE, BATCH_LIST, TOTAL_TRAIN_BATCH) = util.setBatch(1000, 3) # Configurable
total_iteration = 0

with tf.Session() as sess:

    sess.run(init)

    # Try to restore model
    util.debug( util.restoreModel(saver, sess, TRAIN_MODEL_PATH, TRAIN_INFO_PATH) )


    for epoch in range(10000):

        start_time = time.time()

        util.debug("Epoch #" + str(epoch))

        DEBUG_EPOCH = (epoch, DEBUG_EPOCH_INTERVAL)

        X = train_data_array
        y = train_label_onehot

        util.debug("Shuffle and get batch", DEBUG_EPOCH)
        shuffle_order = np.arange(TOTAL_TRAIN_DATA)
        np.random.shuffle(shuffle_order)
        X_shuffle = X[shuffle_order]
        X_batch = np.split(X_shuffle, BATCH_LIST) # batch size = (1000, 1000, 1000, 1728)
        y_shuffle = y[shuffle_order]
        y_batch = np.split(y_shuffle, BATCH_LIST) # batch size = (1000, 1000, 1000, 1728)

        for iteration in range(TOTAL_TRAIN_BATCH):

            DEBUG_ITERATION = (total_iteration, DEBUG_EPOCH_INTERVAL * TOTAL_TRAIN_BATCH)

            # Data Augmentation not required to get good accuracy
            # Data Augmentation
            # util.debug("Data augmentation and preprocessing", DEBUG_ITERATION)
            # operaion = np.random.randint(TOTAL_AUGMENTATION_METHOD, size=(BATCH_SIZE))
            # for data_number in range(BATCH_SIZE):
            #     data = X_batch[iteration][data_number]
            #     method_code = operaion[data_number]
            #     X_batch[iteration][data_number] = AUGMENTATION_METHOD[method_code](data)


            util.debug("Start training", DEBUG_ITERATION)

            (_, train_cost) = sess.run([train_step, cost], feed_dict = {
                                                                image_batch: X_batch[iteration],
                                                                correct_label_onehot: y_batch[iteration],
                                                                is_training: True
                                                            })

            sess.run(reset_vars_initializer)
            sess.run([update_op_accuracy, update_op_precision, update_op_recall], feed_dict = {
                                                                                        image_batch: X_batch[iteration],
                                                                                        correct_label_onehot: y_batch[iteration],
                                                                                        is_training: False
                                                                                    })

            (train_accuracy, train_precision, train_recall) = sess.run([accuracy, precision, recall], feed_dict = {
                                                                                                            image_batch: X_batch[iteration],
                                                                                                            correct_label_onehot: y_batch[iteration],
                                                                                                            is_training: False
                                                                                                        })


            util.debug("Predict Train Data Accuracy = " + str(train_accuracy), DEBUG_ITERATION)
            util.debug("Predict Train Data Precision = " + str(train_precision), DEBUG_ITERATION)
            util.debug("Predict Train Data Recall = " + str(train_recall), DEBUG_ITERATION)
            util.debug("Cost: " + str(train_cost), DEBUG_ITERATION)

            util.debug("Iteration Complete", DEBUG_ITERATION)

            total_iteration = total_iteration + 1

        sess.run(reset_vars_initializer)
        sess.run([update_op_accuracy, update_op_precision, update_op_recall], feed_dict = {
                                                                                    image_batch: test_data_array,
                                                                                    correct_label_onehot: test_label_onehot,
                                                                                    is_training: False
                                                                                })

        (test_accuracy, test_precision, test_recall) = sess.run([accuracy, precision, recall], feed_dict={
                                                                                                    image_batch: test_data_array,
                                                                                                    correct_label_onehot: test_label_onehot,
                                                                                                    is_training: False
                                                                                                })

        util.debug("Predict Test Data Accuracy = " + str(test_accuracy), DEBUG_EPOCH)
        util.debug("Predict Test Data Precision = " + str(test_precision), DEBUG_EPOCH)
        util.debug("Predict Test Data Recall = " + str(test_recall), DEBUG_EPOCH)

        util.debug("Save Model", DEBUG_EPOCH)
        util.saveModel(saver, sess, TRAIN_MODEL_PATH, DEBUG_EPOCH)
        util.writeInfo(TRAIN_INFO_PATH, DEBUG_EPOCH)

        util.debug("Time for epoch " + str(epoch) + " = " + str(time.time() - start_time) + " seconds", DEBUG_EPOCH)
        util.debug("Total program time = " + str((time.time() - PROGRAM_START_TIME)/60) + " minutes", DEBUG_EPOCH)
