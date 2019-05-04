from delaunay import Delaunay2d
import random
random.seed(1234)
import numpy
xyPoints = [numpy.array([random.random(), random.random()]) for i in range(3)]
delaunay = Delaunay2d(xyPoints)

print(delaunay.getEdges())