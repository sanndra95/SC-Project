__author__ = 'Sandra'

import csv
import re
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from keras.layers import Dense,  Dropout
import numpy


DATA = []
EMOTIONS = []

DATA_TEST = []
EMOTIONS_TEST = []

DATA2 = []
EMOTIONS2 = []

DATA_TEST2 = []
EMOTIONS_TEST2 = []

DATA3 = []
EMOTIONS3 = []

DATA_TEST3 = []
EMOTIONS_TEST3 = []

SIZE = 0

NUM_OF_CLASSES = 7

TRAINING_MODE = True


def check_mode():
    return ['"0"', '"1"', '"2"', '"3"', '"4"', '"5"', '"6"']


def load_from_csv(name, d, e):
    happy = 0
    sad = 0
    angry = 0
    neutral = 0
    fear = 0
    surprise = 0
    disgust = 0
    print "Patience..."
    with open(name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            list = []
            row = row[0].split(',')
            checker = check_mode()
            if row[len(row)-1] in checker:
                for r in row[0:len(row)-1]:
                    rr = re.sub('"','',r)
                    new_r = float(rr)
                    list.append(new_r)
                em = re.sub('"','',row[len(row)-1])
                if em == "0":
                    happy += 1
                elif em == "1":
                    sad += 1
                elif em == "2":
                    angry += 1
                elif em == "3":
                    neutral += 1
                elif em == "4":
                    fear += 1
                elif em == "5":
                    surprise += 1
                elif em == "6":
                    disgust += 1
                e.append(float(em))
                d.append(list)

    print "Total: Happy: {}, Sad: {}, Angry: {}, Neutral: {}, Fear: {}, Surprise: {}, Disgust: {}".format(happy, sad, angry, neutral, fear, surprise, disgust)

    return d, e


def train_nn(name):
    x_train1, y_train1 = load_from_csv('D:/Sandra/Faks/4/Soft/soft_datasets/dataset_selenium.csv', DATA, EMOTIONS)
    x_test1, y_test1 = load_from_csv('D:/Sandra/Faks/4/Soft/soft_datasets/testdata_selenium.csv', DATA_TEST, EMOTIONS_TEST)

    x_train2, y_train2 = load_from_csv('D:/Sandra/Faks/4/Soft/soft_datasets/dataset_emoji.csv', DATA2, EMOTIONS2)
    x_test2, y_test2 = load_from_csv('D:/Sandra/Faks/4/Soft/soft_datasets/testdata_emoji.csv', DATA_TEST2, EMOTIONS_TEST2)

    #x_train3, y_train3 = load_from_csv('D:/Sandra/Faks/4/Soft/soft_datasets/dataset.csv', DATA3, EMOTIONS3)
    #x_test3, y_test3 = load_from_csv('D:/Sandra/Faks/4/Soft/soft_datasets/testdata.csv', DATA_TEST3, EMOTIONS_TEST3)

    x_train = x_train1 + x_train2
    y_train = y_train1 + y_train2
    x_test = x_test1 + x_test2
    y_test = y_test1 + y_test2

    print len(x_train), len(y_train), len(x_test), len(y_test)

    global SIZE
    SIZE = len(x_test)

    y_train = to_categorical(y_train, NUM_OF_CLASSES)
    y_test = to_categorical(y_test, NUM_OF_CLASSES)

    numpy.random.seed(7)

    model = Sequential()
    model.add(Dense(137, input_dim=136, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(164, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_OF_CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print "Treniranje..."
    model.fit(x_train, y_train, batch_size=50, epochs=500, shuffle=True, validation_data=(x_test, y_test))

    print "Evaluacija..."
    scores = model.evaluate(x_test, y_test, batch_size=50)
    print "{} ----- {}".format(model.metrics_names, scores)

    print "Predikcija..."
    predictions = model.predict(x_test)
    print predictions
    list = []
    for p in predictions:
        list.append(numpy.argmax(p))

    list2 = []
    happy = 0
    sad = 0
    angry = 0
    neutral = 0
    fear = 0
    surprise = 0
    disgust = 0
    for i in range(len(list)):
        if list[i] == numpy.argmax(y_test[i]):
            list2.append("true")
            if numpy.argmax(y_test[i]) == 0:
                happy += 1
            elif numpy.argmax(y_test[i]) == 1:
                sad += 1
            elif numpy.argmax(y_test[i]) == 2:
                angry += 1
            elif numpy.argmax(y_test[i]) == 3:
                neutral += 1
            elif numpy.argmax(y_test[i]) == 4:
                fear += 1
            elif numpy.argmax(y_test[i]) == 5:
                surprise += 1
            elif numpy.argmax(y_test[i]) == 6:
                disgust += 1

        else:
            list2.append("false")

    print "True: {}".format(list2.count("true"))
    print "False: {}".format(list2.count("false"))

    print "Procenat tacnosti: {}".format((list2.count("true")*100)/SIZE)

    print "Found: Happy: {}, Sad: {}, Angry: {}, Neutral: {}, Fear: {}, Surprise: {}, Disgust: {}".format(happy, sad, angry, neutral, fear, surprise, disgust)

    model.save(name)

def load(name):
    numpy.random.seed(7)

    model = load_model(name)
    print "Loaded model"
    x_test1, y_test1 = load_from_csv('D:/Sandra/Faks/4/Soft/soft_datasets/testdata_selenium.csv', DATA_TEST, EMOTIONS_TEST)
    x_test2, y_test2 = load_from_csv('D:/Sandra/Faks/4/Soft/soft_datasets/testdata_emoji.csv', DATA_TEST2, EMOTIONS_TEST2)

    x_test = x_test1 + x_test2
    y_test = y_test1 + y_test2

    global SIZE
    SIZE = len(x_test)

    y_test = to_categorical(y_test, NUM_OF_CLASSES)


    print "Evaluating..."
    scores = model.evaluate(x_test, y_test, batch_size=50)
    print "{} ----- {}".format(model.metrics_names, scores)

    print "Predikcija..."
    predictions = model.predict(x_test)
    print predictions
    list = []
    for p in predictions:
        list.append(numpy.argmax(p))

    list2 = []
    happy = 0
    sad = 0
    angry = 0
    neutral = 0
    fear = 0
    surprise = 0
    disgust = 0
    for i in range(len(list)):
        if list[i] == numpy.argmax(y_test[i]):
            list2.append("true")
            if numpy.argmax(y_test[i]) == 0:
                happy += 1
            elif numpy.argmax(y_test[i]) == 1:
                sad += 1
            elif numpy.argmax(y_test[i]) == 2:
                angry += 1
            elif numpy.argmax(y_test[i]) == 3:
                neutral += 1
            elif numpy.argmax(y_test[i]) == 4:
                fear += 1
            elif numpy.argmax(y_test[i]) == 5:
                surprise += 1
            elif numpy.argmax(y_test[i]) == 6:
                disgust += 1
        else:
            list2.append("false")

    print "True: {}".format(list2.count("true"))
    print "False: {}".format(list2.count("false"))

    print "Procenat tacnosti: {}".format((list2.count("true") * 100) / SIZE)

    print "Found: Happy: {}, Sad: {}, Angry: {}, Neutral: {}, Fear: {}, Surprise: {}, Disgust: {}".format(happy, sad, angry, neutral, fear, surprise, disgust)

    return model


def start_function():
    if TRAINING_MODE:
        print "Training..."
        train_nn("model23.h5")
    else:
        print "Testing..."
        return load("model23.h5")

if __name__ == "__main__":
    start_function()
