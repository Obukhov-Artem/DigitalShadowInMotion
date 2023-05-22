import numpy as np
from keras import models
from tensorflow import keras
from keras.layers import Dense, Dropout, Input
import math

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class UnitySequence(keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]

        return np.array(batch_x), np.array(batch_y)


class Network:
    def __init__(self):
        self.model = None
        self.result = None

    def CreateModel(self, in_shape, out_shape, neurons=128):

        print("CREATE Network")

        input_layer_data = Input(shape=in_shape)
        h = Dense(neurons, activation="relu")(input_layer_data)

        h = Dense(neurons, activation="relu")(h)
        h = Dropout(0.2)(h)
        #h = Dense(neurons, activation="relu")(h)
        h = Dense(neurons*2, activation="relu")(h)
        h = Dropout(0.2)(h)
        #h = Dense(neurons * 2, activation="relu")(h)
        h = Dense(neurons * 4, activation="linear")(h)

        output_layer = Dense(out_shape, name="output")(h)

        self.model = models.Model(input_layer_data, output_layer,
                                  name="SIMPLE")
        self.model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["accuracy"],

        )


        self.model.summary()

        return self.model

    def TrainGen(self, generator, test_generator, epoch=10):

        self.result = self.model.fit(generator,
                                     epochs=epoch,
                                     validation_data=test_generator,
                                     shuffle=True,
                                     verbose=1
                                     )

        return self.result

    def __Format(self, data):
        dataTemp = []
        for i in range(0, len(data)):
            dataTemp.append([])
            for j in range(0, len(data[i])):
                dataTemp[i].append(float(data[i][j].replace(',', '.')))
        return np.array(dataTemp)

    def Result(self):
        return (self.result.history["accuracy"])

    def Predict(self, input):
        return self.model.predict(input, verbose=0)

    def Save(self, filepath):
        self.model.save(filepath)

    def Load(self, filepath):
        self.model = models.load_model(filepath)
        return self.model

