import pandas as pn
from os import listdir
from os.path import isfile, join

import Enum.BodyPart
from Enum import BodyPart

class CSVReader():
    _allFile: list = None

    def __init__(self):
        self._allFile = []

    def Read(self, pathDir,header=[1,2,3]):

        for fileName in listdir(pathDir):
            if isfile(join(pathDir, fileName)):
                self._allFile.append(pn.read_csv(pathDir+fileName,header=header, sep=';'))
        self._allFile = pn.concat(self._allFile).iloc[:,:55]
        return self._allFile

    def SelectionName(self, *col,rotation=False):
        """

        :param col:
        :param rotation:
        :return: numpy array values
        """

        selDataFrame = sum([i if type(i) is list else [float(i)] for i in col], [])
        if rotation:
            selDataFrame = sum([i.coord_with_roration() for i in selDataFrame], [])
        else:
            selDataFrame = sum([i.coord() for i in selDataFrame], [])
        selDataFrame.sort()
        resDataFrame = self._allFile.take(selDataFrame, axis=1)
        return resDataFrame.values
    def SelectionIndex(self, *col):
        #useless now
        #print(list(col) if (type(col[0]) is not list) else sum([i for i in col],[])) # this moster is a live! LIVE!!!
        selDataFrame = sum([i if type(i) is list else [i] for i in col ], [])
        selDataFrame = sum([[3*i,3*i+1,3*i+2] for i in selDataFrame], [])
        selDataFrame.sort()
        resDataFrame = self._allFile.take(selDataFrame, axis=1)
        return resDataFrame
    def SelectionExcludeName(self, *col):
        selDataFrame = sum([i.coord() for i in col],[])
        bodyPoint = BodyPart.coord()
        selDataFrame.sort()
        selDataFrame = [i for i in bodyPoint if i not in selDataFrame]
        resDataFrame = self._allFile.take(selDataFrame, axis=1)
        return resDataFrame
    def SelectionExcludeIndex(self, *col):
        selDataFrame = sum([i if type(i) is list else [i] for i in col ], [])
        selDataFrame = sum([[3 * i, 3 * i + 1, 3 * i + 2] for i in selDataFrame], [])
        selDataFrame.sort()
        bodyPoint = BodyPart.coord()
        selDataFrame = [i for i in bodyPoint if i not in selDataFrame]
        resDataFrame = self._allFile.take(selDataFrame, axis=1)
        return resDataFrame

