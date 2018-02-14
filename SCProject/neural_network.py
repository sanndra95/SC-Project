__author__ = 'Sandra'

import csv
import re
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Activation, Dropout
import time
import numpy
import h5py

DATA = []
EMOTIONS = []

DATA_TEST = []
EMOTIONS_TEST = []

SIZE = 0

NUM_OF_CLASSES = 4

SAD_MODE = False
TRAINING_MODE = False

MODEL_NAME = "model.h5"
MODEL_NAME_WITHOUT_SAD = "model_ws.h5"

def check_mode():
    if SAD_MODE:
        return ['"0"', '"2"', '"3"']
    else:
        return ['"0"', '"1"', '"2"', '"3"']


def load_from_csv(name, d, e):
    happy = 0
    sad = 0
    angry = 0
    neutral = 0
    print "Patience..."
    with open(name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            list = []
            row = row[0].split(',')
            #print "Emocija: {}".format(row[len(row)-1])
            checker = check_mode()
            #print checker
            if row[len(row)-1] in checker:
                for r in row[0:len(row)-1]:
                    rr = re.sub('"','',r)
                    new_r = float(rr)
                    #print len(row)
                    list.append(new_r)
                    #print "emocija: {}".format(row[len(row)-1])
                #print len(list)
                em = re.sub('"','',row[len(row)-1])
                #print em
                if em == "0":
                    happy += 1
                elif em == "1":
                    sad += 1
                elif em == "2":
                    angry += 1
                elif em == "3":
                    neutral += 1
                e.append(float(em))
                d.append(list)

    print d[0]
    print e[0]
    print "Total: Happy: {}, Sad: {}, Angry: {}, Neutral: {}".format(happy, sad, angry, neutral)

    return d, e


def train_nn(name):
    x_train, y_train = load_from_csv('dataset.csv', DATA, EMOTIONS)
    x_test, y_test = load_from_csv('testdata.csv', DATA_TEST, EMOTIONS_TEST)

    print len(x_train), len(y_train), len(x_test), len(y_test)

    global SIZE
    SIZE = len(x_test)

    #print x_train[0], y_train[0], x_test[0], y_test[0]

    print y_train
    print y_test
    #prebacivanje emocija ako nema sad:
    #sad je broj 1, ako nema, angry postaje 1 umesto 2
    #neutral postaje 2 umesto 3

    if SAD_MODE:
        for n, i in enumerate(y_train):
            if i == 2.0:
                y_train[n] = 1.0
            elif i == 3.0:
                y_train[n] = 2.0

        for n, i in enumerate(y_test):
            if i == 2.0:
                y_test[n] = 1.0
            elif i == 3.0:
                y_test[n] = 2.0

        print y_train
        print y_test



    y_train = to_categorical(y_train, NUM_OF_CLASSES)
    y_test = to_categorical(y_test, NUM_OF_CLASSES)

    #print x_train.shape
    #print x_test.shape
    print y_train.shape
    print y_test.shape

    numpy.random.seed(7)

    model = Sequential()
    model.add(Dense(32, input_dim=136, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(16, activation='relu'))
    #model.add(Dense(100, activation='relu'))
    model.add(Dense(NUM_OF_CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print "Treniranje..."
    model.fit(x_train, y_train, batch_size=10, epochs=150, shuffle=False, validation_data=(x_test, y_test))

    #model.save('model.h5')

    print "Evaluacija..."
    scores = model.evaluate(x_test, y_test, batch_size=10)
    print "{} ----- {}".format(model.metrics_names, scores)

    print "Predikcija..."
    predictions = model.predict(x_test)
    #for x in predictions:
        #print round(max(x))
    #for x in predictions:
        #print x
    print predictions
    list = []
    #f = open('predictions.txt', 'ab')
    for p in predictions:
        list.append(numpy.argmax(p))

    list2 = []
    happy = 0
    sad = 0
    angry = 0
    neutral = 0
    for i in range(len(list)):
        if list[i] == numpy.argmax(y_test[i]):
            list2.append("true")
            if numpy.argmax(y_test[i]) == 0:
                happy += 1
            elif numpy.argmax(y_test[i]) == 1:
                if SAD_MODE:
                    angry += 1
                else:
                    sad += 1
            elif numpy.argmax(y_test[i]) == 2:
                if SAD_MODE:
                    neutral += 1
                else:
                    angry += 1
            elif numpy.argmax(y_test[i]) == 3:
                if not SAD_MODE:
                    neutral += 1
        else:
            list2.append("false")
        #print list[i]
        #print y_test[i]
        #f.write(str(i))
        #f.write("\n")
    #f.close()

    print "True: {}".format(list2.count("true"))
    print "False: {}".format(list2.count("false"))

    print "Procenat tacnosti: {}".format((list2.count("true")*100)/SIZE)

    print "Found: Happy: {}, Sad: {}, Angry: {}, Neutral: {}".format(happy, sad, angry, neutral)

    model.save(name)

def load(name):
    numpy.random.seed(7)

    model = load_model(name)
    print "Loaded model"
    x_test, y_test = load_from_csv('testdata.csv', DATA_TEST, EMOTIONS_TEST)

    global SIZE
    SIZE = len(x_test)

    #print x_test
    #print y_test

    if SAD_MODE:
        for n, i in enumerate(y_test):
            if i == 2.0:
                y_test[n] = 1.0
            elif i == 3.0:
                y_test[n] = 2.0

    y_test = to_categorical(y_test, NUM_OF_CLASSES)

    #print y_test

    #print y_test.shape
    #print len(x_test)

    print "Evaluating..."
    scores = model.evaluate(x_test, y_test, batch_size=10)
    print "{} ----- {}".format(model.metrics_names, scores)

    print "Predikcija..."
    predictions = model.predict(x_test)
    # for x in predictions:
    # print round(max(x))
    # for x in predictions:
    # print x
    print predictions
    list = []
    # f = open('predictions.txt', 'ab')
    for p in predictions:
        list.append(numpy.argmax(p))

    list2 = []
    happy = 0
    sad = 0
    angry = 0
    neutral = 0
    for i in range(len(list)):
        if list[i] == numpy.argmax(y_test[i]):
            list2.append("true")
            if numpy.argmax(y_test[i]) == 0:
                happy += 1
            elif numpy.argmax(y_test[i]) == 1:
                if SAD_MODE:
                    angry += 1
                else:
                    sad += 1
            elif numpy.argmax(y_test[i]) == 2:
                if SAD_MODE:
                    neutral += 1
                else:
                    angry += 1
            elif numpy.argmax(y_test[i]) == 3:
                if not SAD_MODE:
                    neutral += 1
        else:
            list2.append("false")
            # print list[i]
            # print y_test[i]
            # f.write(str(i))
            # f.write("\n")
    # f.close()

    print "True: {}".format(list2.count("true"))
    print "False: {}".format(list2.count("false"))

    print "Procenat tacnosti: {}".format((list2.count("true") * 100) / SIZE)

    print "Found: Happy: {}, Sad: {}, Angry: {}, Neutral: {}".format(happy, sad, angry, neutral)


    return model

def start_function():
    if TRAINING_MODE and SAD_MODE:
        print "1111111"
        train_nn("model_ws.h5")
    elif TRAINING_MODE and not SAD_MODE:
        print "2222222"
        train_nn("model.h5")
    elif not TRAINING_MODE and SAD_MODE:
        print "3333333"
        return load("model_ws.h5")
    elif not TRAINING_MODE and not SAD_MODE:
        print "4444444"
        return load("model.h5")

if __name__ == "__main__":
    start_function()
