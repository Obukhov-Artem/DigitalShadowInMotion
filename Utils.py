def MSD(pointOne,pointTwo): #MeanSquareDeviation
    allMSD = []
    for i in range(0, len(pointOne), 3):
        allMSD.append(((pointOne[i] - pointTwo[i]) ** 2 + (pointOne[i + 1] - pointTwo[i + 1]) ** 2 + (pointOne[i + 2] - pointTwo[i + 2]) ** 2) ** (1 / 2))
    return allMSD

