import sys
import tensorflow as tf
import csv
import numpy as np


def loadData(train_path, test_path):
    kDigitsCount = 10
    train_data = [[],[]]
    with open(train_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        is_header = True
        for row in reader:
            if is_header:
                is_header = False
            else:
                label = np.zeros(kDigitsCount)
                label[int(row[0])] = 1
                train_data[0].append(label)
                data = [int(i) for i in row[1:]]
                train_data[1].append(data)
    train_data[1] = np.array(train_data[1])
    train_data[0] = np.array(train_data[0])
    print(train_data[0].shape, train_data[1].shape)

    test_data = []
    with open(test_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        is_header = True
        for row in reader:
            if is_header:
                is_header = False
            else:
                data = [int(i) for i in row[0 : ]]
                test_data.append(data)
    test_data = np.array(test_data)
    return train_data, test_data


def rand_choice(elem_count, arrays):
    if len(arrays) == 0:
        return []
    idx = np.random.randint(arrays[0].shape[0], size=elem_count)
    res = []
    for array in arrays:
        res.append(array[idx])
    return res


class MnistModel:
    def __init__(self):
        self.kDigitsCount = 10
        self.kPixelsCount = 784
        
        # y = x*W + b - predict
        self.x = tf.placeholder(tf.float32, [None, self.kPixelsCount])
        self.W = tf.Variable(tf.zeros([self.kPixelsCount, self.kDigitsCount]))
        self.b = tf.Variable(tf.zeros([self.kDigitsCount]))
        self.y = tf.matmul(self.x, self.W) + self.b
        # y_ - true value
        self.y_ = tf.placeholder(tf.float32, [None, self.kDigitsCount])

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_))
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)

        self.predict_label = tf.argmax(self.y, 1)

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
  
        self.sess = tf.InteractiveSession()


    def predict(self, test_data):
        # format output: digit
        return self.sess.run(self.predict_label, feed_dict= {self.x: test_data})
    

    def check(self, data, labels):
        # label must be array like [0, ..., 1, ..., 0]
        return self.sess.run(self.accuracy, feed_dict={self.x: data, self.y_: labels})


    def train(self, train_data, batch_size, it_count):
        print("Start train:")
        tf.initialize_all_variables().run()
        for it in range(it_count):
            print("it:", it + 1, "/", it_count)
            batch_ys, batch_xs = rand_choice(batch_size, train_data)
            self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})


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
                        np.concatenate((start_part1, end_part1), axis=0)], 100, 2000)
            score += self.check(middle_part1, middle_part0)
        score /= parts_count
        return score


def save_predictions(predictions, path):
    #save predictions in csv file
    with open(path, 'w') as csvfile:
        fieldnames = ['ImageId', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, label in enumerate(predictions):
            writer.writerow({'ImageId': i + 1, 'Label': label})


def main(_):
    train_data, test_data = loadData('data/train.csv','data/test.csv')
    model = MnistModel()
    #model.train(train_data, 100, 1000)
    #predictions = model.predict(test_data)
    #save_predictions(predictions, sys.argv[1])
    value = model.cross_validation(train_data)
    print(value)
    

if __name__ == '__main__':
    tf.app.run()