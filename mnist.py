import sys
import tensorflow as tf
import csv
import numpy as np
from model import *


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
    if sys.argv[1] == "cv":
        value = model.cross_validation(train_data)
        print(value)
    elif sys.argv[1] == "gen":
        model.train(train_data, 50, 4000)
        predictions = model.predict(test_data)
        save_predictions(predictions, sys.argv[2])
    

if __name__ == '__main__':
    tf.app.run()