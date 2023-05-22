import os
import socket

from CSVRead import CSVReader
from Enum import BodyPart
from NeuNetwork import Network

order3 = [BodyPart.head.head, BodyPart.ArmRight.armLower, BodyPart.ArmLeft.armLower]
order5 = [BodyPart.head.head, BodyPart.ArmRight.armLower, BodyPart.ArmLeft.armLower, BodyPart.FootRight.snkle, BodyPart.FootLeft.snkle]

patchModel = os.path.dirname(__file__) + "/models/"
pathDirControl = os.path.dirname(__file__) + "/test/"
adress = "127.0.0.1"
port = 8100
step = 5
modelVersion = 5

if( modelVersion == 3):
    order = order3
elif(modelVersion == 5):
    order = order5

Net = Network()
Net.Load(patchModel + "full_model" + str(modelVersion) + ".h5")

csvReaderControl = CSVReader()
dataSetControl = csvReaderControl.Read(pathDirControl)
inputs = csvReaderControl.SelectionName(order)

run = True
while (run):

    _socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    f = _socket.connect_ex((adress, port))
    index = 0
    try:
        while (True):

            if (index >= len(inputs)):
                index = 0
            primer = []
            for i in range(0, len(inputs[index])):
                primer.append(inputs[index][i])

            Prediction = Net.Predict([primer])[0]
            send = []
            for i in range(len(Prediction)):
                send.append(Prediction[i])

            _socket.send(bytes(str(send)[1:-2].replace(",", "") + '\n', 'ascii'))
            index += step
    except:
        print("Соединение прерванно")
