class D(): # dimension
    one = 0
    two = 1
    three = 2
    four = 3

def Point(i):
    return[0+i,1+i,2+i]

class Coord():
    def __init__(self,i, j):
        self._i = i
        self._j = j
        self.x = i + self._j * 3 + D.one
        self.y = i + self._j * 3 + D.two
        self.z = i + self._j * 3 + D.three


    def coord(self):
        return [self.x, self.y, self.z]
    def coord_with_roration(self):
        return [self.x, self.y, self.z, self.x+54, self.y+54, self.z+54]
    def __str__(self):
        return str([self.x, self.y, self.z])
    def id(self):
        return self._i + self._j

class __Head():
    def __init__(self,i):
        self._id = i
        self.head = Coord(3*i, 0)

    def coord(self):
        return self.head.coord()
    def __str__(self):
        return str(self.coord())
    def id(self):
        return self._id

class __Spine():
    def __init__(self,i):
        self._id = i
        self.spineUpper = Coord(9*i-6,0)
        self.spineMiddle = Coord(9*i-6, 1)
        self.spineLower = Coord(9*i-6, 2)

    def coord(self):
        return self.spineUpper.coord() + self.spineMiddle.coord() + self.spineLower.coord()
    def __str__(self):
        return str(self.coord())
    def id(self):
        return self._id


class __Arm():
    def __init__(self,i):
        self._id = i
        self.armUpper = Coord(9*i-6, 0)
        self.armMiddle = Coord(9*i-6, 1)
        self.armLower = Coord(9*i-6, 2)

    def coord(self):
        return self.armUpper.coord() + self.armMiddle.coord() + self.armLower.coord()
    def __str__(self):
        return str(self.coord())
    def id(self):
        return self._id

class __Foot():
    def __init__(self,i):
        self._id = i
        self.hip = Coord(12*i-18, 0)
        self.knee = Coord(12*i-18, 1)
        self.snkle = Coord(12*i-18, 2)
        self.toe = Coord(12*i-18, 3)
    def coord(self):
        return self.hip.coord() + self.knee.coord() + self.snkle.coord() + self.toe.coord()
    def __str__(self):
        return str(self.coord())
    def id(self):
        return self._id


head = __Head(0)
spine = __Spine(1)
ArmRight = __Arm(2)
ArmLeft = __Arm(3)
FootRight = __Foot(4)
FootLeft = __Foot(5)

def coord():
    return head.coord() + spine.coord() + ArmRight.coord()\
        + ArmLeft.coord() + FootRight.coord() + FootLeft.coord()
def Name():
    return [head,spine,ArmRight, ArmLeft, FootRight, FootLeft]

def string():
    return str(coord())