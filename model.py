import numpy as np
import tensorflow as tf


def rand_choice(elem_count, arrays):
    if len(arrays) == 0:
        return []
    idx = np.random.randint(arrays[0].shape[0], size=elem_count)
    res = []
    for array in arrays:
        res.append(array[idx])
    return res


def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class MnistModel:
    def __init__(self):
        self.kDigitsCount = 10
        self.kImageHeight = 28
        self.kImageWidth = 28
        
        #prepare data
        self.x = tf.placeholder(tf.float32, [None, self.kImageHeight * self.kImageWidth])
        self.x_image = tf.reshape(self.x, [-1,28,28,1])
        # y_ - true value
        self.y_ = tf.placeholder(tf.float32, [None, self.kDigitsCount])

        # First conv layer
        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])
        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)

        # Second conv layer
        self.W_conv2 = weight_variable([5, 5, 32, 64])
        self.b_conv2 = bias_variable([64])
        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = max_pool_2x2(self.h_conv2)

        # Densely connected layer
        self.W_fc1 = weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = bias_variable([1024])
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        # Dropout
        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
        
        # Readout Layer
        self.W_fc2 = weight_variable([1024, self.kDigitsCount])
        self.b_fc2 = bias_variable([self.kDigitsCount])
        self.y = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2

        # Optimize
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        # Get label
        self.predict_label = tf.argmax(self.y, 1)

        # Check
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
  
        self.sess = tf.InteractiveSession()


    def predict(self, test_data):
        # format output: digit
        kBatchSize = min(100, test_data.shape[0])
        it_count = test_data.shape[0] // kBatchSize
        res = []
        for it in range(it_count):
            print("It predict:", it)
            res.append(
                self.sess.run(self.predict_label, 
                              feed_dict= {
                                          self.x: test_data[it * kBatchSize : (it + 1) * kBatchSize], 
                                          self.keep_prob: 1.0
                                         }
                             )
                )
        return np.concatenate(res, axis=0)
    

    def check(self, check_data):
        # label must be array like [0, ..., 1, ..., 0]
        kBatchSize = min(100, check_data[0].shape[0])
        it_count = check_data[0].shape[0] // kBatchSize
        score = 0
        for it in range(it_count):
            score += self.sess.run(self.accuracy, 
                                   feed_dict= {
                                               self.x: check_data[1][it * kBatchSize : (it + 1) * kBatchSize], 
                                               self.y_: check_data[0][it * kBatchSize : (it + 1) * kBatchSize],
                                               self.keep_prob: 1.0
                                              }
                                    )
        return score / it_count


    def train(self, train_data, batch_size, it_count):
        print("Start train:")
        tf.initialize_all_variables().run()
        for it in range(1, it_count + 1):
            print("it:", it, "/", it_count)
            batch_ys, batch_xs = rand_choice(batch_size, train_data)
            if it % 250 == 0:
               print("Accuracy:", self.check([batch_ys, batch_xs]))
            self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 0.5})


    def cross_validation(self, train_data):
        #divide into 10 parts
        parts_count = 10
        part_size = len(train_data[0]) // parts_count
        score = 0
        for it in range(0, parts_count):
            print("CC it:", it + 1, '/', parts_count)
            start_part0 = train_data[0][0 : it * part_size]
            middle_part0 = train_data[0][it * part_size : (it + 1) * part_size]
            end_part0 = train_data[0][(it + 1) * part_size : parts_count * part_size] 

            start_part1 = train_data[1][0 : it * part_size]
            middle_part1 = train_data[1][it * part_size : (it + 1) * part_size]
            end_part1 = train_data[1][(it + 1) * part_size : parts_count * part_size] 

            self.train([np.concatenate((start_part0, end_part0), axis=0), 
                        np.concatenate((start_part1, end_part1), axis=0)], 50, 1000)
            score += self.check([middle_part0, middle_part1])
        score /= parts_count
        return score
