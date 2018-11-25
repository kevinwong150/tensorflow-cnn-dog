import os
import util
import numpy as np
import math
import tensorflow as tf
from operator import sub, mul

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
RANDOM_SEED = 1

data = np.arange(1, 100 + 1)
data_input = tf.constant(data)

batch_shuffle = tf.train.shuffle_batch([data_input], enqueue_many=True, batch_size=23, capacity=100, min_after_dequeue=10, allow_smaller_final_batch=True)
batch_no_shuffle = tf.train.batch([data_input], enqueue_many=True, batch_size=23, capacity=100, allow_smaller_final_batch=True)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # for i in range(5):
    print(sess.run([batch_no_shuffle]))
    print(sess.run([batch_no_shuffle]))
    print(sess.run([batch_no_shuffle]))
    print(sess.run([batch_no_shuffle]))
    print(sess.run([batch_no_shuffle]))

    coord.request_stop()
    coord.join(threads)
