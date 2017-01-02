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


def run(train_data, test_data):
    kBatchSize = 100
    kDigitsCount = 10
    kPixelsCount = 784
    # Model
    x = tf.placeholder(tf.float32, [None, kPixelsCount])
    W = tf.Variable(tf.zeros([kPixelsCount, kDigitsCount]))
    b = tf.Variable(tf.zeros([kDigitsCount]))
    y = tf.matmul(x, W) + b
    # Loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, kDigitsCount])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    # Train
    print('start_train')
    cur_pos = 0
    for it in range(420):
        print('it:', it)
        batch_xs = train_data[1][cur_pos:cur_pos + kBatchSize]
        batch_ys = train_data[0][cur_pos:cur_pos + kBatchSize]
        cur_pos += kBatchSize
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    res_y = tf.argmax(y, 1)
    res = sess.run(res_y, feed_dict= {x: test_data})
    print(res)
    return res


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
    predictions = run(train_data, test_data)
    save_predictions(predictions, sys.argv[1])


if __name__ == '__main__':
    tf.app.run()