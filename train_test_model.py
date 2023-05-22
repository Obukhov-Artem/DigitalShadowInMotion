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

flag_save = True
flag_train_simple = False
TRAIN_SIZE = 50001
if flag_train_simple:
    train_x_data = [
        train_dataset.take(get_column_dataset(o, rotation=ROTATION), axis=1).to_numpy()[1:TRAIN_SIZE, :INPUT_SHAPE_DATA] for o in
        order]
    train_y_data = train_dataset.values[1:TRAIN_SIZE, :OUTPUT_SHAPE]

    test_x_data = [test_dataset.take(get_column_dataset(o, rotation=ROTATION), axis=1).to_numpy()[1:, :INPUT_SHAPE_DATA]
                   for o in order]
    test_y_data = test_dataset.values[1:, :OUTPUT_SHAPE]

    print("Prepare train data: ", time.time() - time_start)
    time_start = time.time()

    models = []
    for j in range(len(order)):
        print("CREATE", len(order[j]))
        model = Network()
        training_generator = UnitySequence(train_x_data[j], train_y_data, BATCH)
        test_generator = UnitySequence(test_x_data[j], test_y_data, BATCH)
        model.CreateModel(len(order[j]) * 3,OUTPUT_SHAPE, neurons=NEURON)
        models.append(model.TrainGen(training_generator, test_generator, epoch=EPOCH).history)
        if flag_save:
            model.Save("model" + str(len(order[j])) + ".h5")

print("Train NN: ", time.time() - time_start)
time_start = time.time()
EXAMPLES = 80000


test_x_data = [test_dataset.take(get_column_dataset(o, rotation=ROTATION), axis=1).to_numpy(dtype='float32')[1:,
               :INPUT_SHAPE_DATA]
               for o in order]
test_y_data = test_dataset.values[:, :OUTPUT_SHAPE].astype(np.float32)
model = Network()

models = [model.Load(pathModel + "model" + str(len(order[j])) + ".h5") for j in
             range(len(order))]

print("LOAD")
predict_data = {'NN':[]}
result_eval = []
time_eval = {'NN':[]}
for j in range(len(order)):
    result_simple_time = []
    control_x = test_x_data[j][:EXAMPLES]
    control_y = test_y_data[:EXAMPLES]

    simple_prediction = models[j].predict(control_x)
    print(evaluate_data(str(len(order[j])) + " ", test_y_data[:EXAMPLES], simple_prediction,0))

