import os
import time
import numpy as np
from Enum import BodyPart
from CSVRead import CSVReader
from NeuNetwork import UnitySequence,Network
import datetime
import pandas as pd
import os
import csv

from keras import models
from tensorflow import keras
from keras.layers import Dense, Dropout, Input, LSTM,RNN,Conv1D,Reshape,Flatten
def get_column_dataset(parts, rotation=False):
    if rotation:
        selDataFrame = sum([i.coord_with_roration() for i in parts], [])
    else:
        selDataFrame = sum([i.coord() for i in parts], [])
    selDataFrame.sort()
    return selDataFrame



def evaluate_data(num, etalon, data, timer=0):
    etalon = etalon.reshape((len(etalon), 18, 3))
    data = data.reshape((len(data), 18, 3))
    mse = np.array([np.mean(np.sqrt(np.sum(np.square(etalon[i] - data[i]), axis=1))) for i in range(len(data))])
    mmax = np.array([np.max(np.sqrt(np.sum(np.square(etalon[i] - data[i]), axis=1))) for i in range(len(data))])
    msum = np.array([np.sum(np.sqrt(np.sum(np.square(etalon[i] - data[i]), axis=1))) for i in range(len(data))])
    N = 3
    print(num, str(round(np.mean(mse), N)) + str(" ± ") + str(round(np.std(mse), N)),
          str(round(np.mean(mmax), N)) + str(" ± ") + str(round(np.std(mmax), N)),str(round(np.mean(msum), N)) + str(" ± ") + str(round(np.std(msum), N)),
           str(round(timer*1000, N)), sep="\t")


    return [num, np.mean(mse), np.std(mse), np.mean(mmax), np.std(mmax), timer]

MainPath = ".//Data_Animation/MotionCapture 19.01.23//"
pathDirControl = "./Data_Animation/Test 19.01.23//"
cur_folder = str(datetime.datetime.now()).replace(":", "_")
pathModel = "./Final_model/"

time_start = time.time()

trainReader = CSVReader()
train_dataset = trainReader.Read(MainPath)

testReader = CSVReader()
test_dataset = testReader.Read(pathDirControl,header=None)


print(train_dataset.values.shape, test_dataset.values.shape)
print("Load data: ", time.time() - time_start)
time_start = time.time()
BATCH = 200
EPOCH = 10
NEURON = 400

ROTATION = False
INPUT_SHAPE_DATA = 54
OUTPUT_SHAPE = 54

order = [[BodyPart.head.head, BodyPart.ArmRight.armLower, BodyPart.ArmLeft.armLower],
         [BodyPart.head.head, BodyPart.ArmRight.armLower, BodyPart.ArmLeft.armLower, BodyPart.FootLeft.snkle,
          BodyPart.FootRight.snkle],
         [BodyPart.head.head, BodyPart.ArmRight.armMiddle, BodyPart.ArmRight.armLower, BodyPart.ArmLeft.armMiddle,
          BodyPart.ArmLeft.armLower,
          BodyPart.FootLeft.snkle, BodyPart.FootRight.snkle],
         [BodyPart.head.head, BodyPart.ArmRight.armMiddle, BodyPart.ArmRight.armLower, BodyPart.ArmLeft.armMiddle,
          BodyPart.ArmLeft.armLower,
          BodyPart.FootLeft.knee, BodyPart.FootLeft.snkle,
          BodyPart.FootRight.knee, BodyPart.FootRight.snkle],
         [BodyPart.head.head, BodyPart.ArmRight.armMiddle, BodyPart.ArmRight.armLower, BodyPart.ArmLeft.armMiddle,
          BodyPart.ArmLeft.armLower,
          BodyPart.FootLeft.hip, BodyPart.FootLeft.knee, BodyPart.FootLeft.snkle,
          BodyPart.FootRight.hip, BodyPart.FootRight.knee, BodyPart.FootRight.snkle],
         [BodyPart.head.head, BodyPart.spine.spineUpper, BodyPart.spine.spineLower, BodyPart.ArmRight.armMiddle,
          BodyPart.ArmRight.armLower, BodyPart.ArmLeft.armMiddle,
          BodyPart.ArmLeft.armLower,
          BodyPart.FootLeft.hip, BodyPart.FootLeft.knee, BodyPart.FootLeft.snkle,
          BodyPart.FootRight.hip, BodyPart.FootRight.knee, BodyPart.FootRight.snkle]
         ]

TRAIN_SIZE = 5001
TEST_SIZE = 11
hidden_layer = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
neurons =[20, 50, 100, 200, 400, 800, 1600]
neuron_increment = [True, False]
epochs = [5, 10, 15, 25, 50]
layers = ["Dense", "Conv1D", "RNN", "LSTM"]
dropout_enable = [True, False]
batch_size = 50
in_shape = 9
out_shape = 54

train_x_data = [
        train_dataset.take(get_column_dataset(o, rotation=ROTATION), axis=1).to_numpy()[1:TRAIN_SIZE, :INPUT_SHAPE_DATA] for o in
        order]
train_y_data = train_dataset.values[1:TRAIN_SIZE, :OUTPUT_SHAPE]

test_x_data = [test_dataset.take(get_column_dataset(o, rotation=ROTATION), axis=1).to_numpy()[1:TEST_SIZE, :INPUT_SHAPE_DATA]
               for o in order]
test_y_data = test_dataset.values[1:TEST_SIZE, :OUTPUT_SHAPE]

def train_one_model(hl=1,n=20,n_i=False,e=10,l="Dense",de=False):

    training_generator = UnitySequence(train_x_data[0], train_y_data, batch_size)
    test_generator = UnitySequence(test_x_data[0], test_y_data, batch_size)
    input_layer_data = Input(shape=in_shape)

    h = Dense(n, activation="relu")(input_layer_data)
    for layer in range(1,hl):
        if l == "Dense":
            h = Dense(n, activation="relu")(h)
        elif l == "Conv1D":
            h = Reshape((-1,1))(h)
            h = Conv1D(filters=n,kernel_size=2, activation="relu")(h)
        elif l == "RNN":
            h = Reshape((-1,1))(h)
            h = RNN(n, activation="relu")(h)
        elif l == "LSTM":
            h = Reshape((-1,1))(h)
            h = LSTM(n, activation="relu")(h)
        if de:
            h = Dropout(0.2)(h)
        if n_i:
            n = n*2

    if l != "Dense":
        h = Flatten()(h)
    output_layer = Dense(out_shape, name="output")(h)

    model = models.Model(input_layer_data, output_layer,
                              name="SIMPLE")
    model.compile(
        optimizer="adam",
        loss="mse"

    )
    result = model.fit(training_generator,
                                 epochs=e,
                                 validation_data=test_generator,
                                 shuffle=True,
                                 verbose=0
                                 )
    return result

for de in dropout_enable:
    r = train_one_model(de=de)
    print(de, r.history)

for ni in neuron_increment:
    r = train_one_model(n_i=ni, de=True)
    print(ni, r.history)

for l in layers:
    r = train_one_model(l=l,n_i=True, de=True)
    print(l, r.history)

for h in hidden_layer:
    r = train_one_model(hl=h,l="Dense",n_i=True, de=True)
    print(h,r.history)

for n in neurons:
    r = train_one_model(n=n,l="Dense",hl=5,n_i=True, de=True)
    print(n,r.history)

for e in epochs:
    r = train_one_model(bs=50,e=e,l="Dense",n=500,hl=5,n_i=True, de=True)
    print(r, r.history["loss"])
