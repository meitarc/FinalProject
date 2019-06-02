import cv2
import numpy as np

######### findObjects #########
#Gets:  image - an image as defined by cv2.imgread().
#       prototxtPath, caffemodelPath - the path to the prototxt and caffemodel files needed for the object detection.
#       threshold - the threshold to decide if an object is recognized or not, default should be 0.6
#Returns: a list of (startX, startY, endX, endY) ranges of the objects found.

def findObject(image, prototxtPath, caffemodelPath):
    return findObjectsWithThreshold(image, prototxtPath, caffemodelPath, 0.6)

######### findObjectsWithThreshold #########
# like the findObjects, with the option to define the threshold.
def findObjectsWithThreshold(image, prototxtPath, caffemodelPath, threshold):
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe(prototxtPath, caffemodelPath)
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    # (note: normalization is done via the authors of the MobileNet SSD implementation)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    #print(blob.shape)
    # pass the blob through the network and obtain the detections and predictions
    #print('blob')
    #print(blob[0][0][0][0])
    net.setInput(blob)
    detections = net.forward()
    #print(detections.shape)
    detectionList = []
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        #print('confidence:')
        #print(confidence)
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence >= threshold:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            y = startY - 15 if startY - 15 > 15 else startY + 15
            temp = (startY, endY, startX, endX)
            detectionList.append(temp)

    #print('test: ')
    #print(detectionList)
    return detectionList

#example input:
#imagePath = 'images/carTest.jpg'
#imageFile = cv2.imread(imagePath)
#threshold = 0.6
#prototxtPath = 'MobileNetSSD_deploy.prototxt.txt'
#caffemodelPath = 'MobileNetSSD_deploy.caffemodel'
#
#example output use:
#list = findObjects(imageFile, prototxtPath, caffemodelPath, 0.6)
#
#output:
#list = [(606, 854, 1037, 1482), (524, 807, 596, 976)] #an exmaple output
#cv2.imwrite('test.jpg', imageFile[524:807, 596:976]) #an example saving specific object into a unique image

######### returnSubImage( image, startY, endY, startX, endX) #########
def returnSubImage( image, startY, endY, startX, endX):
    temp = image.copy()
    temp = temp[startY:endY, startX:endX]
    return temp

#MAIN TEST
from functions import funcCheck1, funcCheck2

prototxtPath = 'MobileNetSSD_deploy.prototxt.txt'
caffemodelPath = 'MobileNetSSD_deploy.caffemodel'
#opening Image Example
image1 = cv2.imread('source/115.jpg')
#cv2.imwrite('output/ObjectDetectionsResults/test1.jpg', image1)
#finding the objects
objects = findObjectsWithThreshold(image1, prototxtPath, caffemodelPath, 0.1)
print('objects: ')
print(objects)
#firstObject = objects[0]
#print(firstObject)
#saving the first object as a seperate image
#image2 = returnSubImage(image1, objects[0][0], objects[0][1], objects[0][2], objects[0][3])
#image2 = returnSubImage(image1, 531,681,873,1105)
image2 = image1.copy()
image3 = image2[580:888 , 1236:1780]
image4 = image2[500:844 , 735:1189]
image5 = image2[475:747 , 634:740]

#p,okp,odes = funcCheck1(image1,image1)
#img2 = cv2.drawKeypoints(image2,okp, image2)
cv2.imwrite('output/ObjectDetectionsResults/original Image.jpg', image1)
cv2.imwrite('output/ObjectDetectionsResults/found object.jpg', image2)
cv2.imwrite('output/ObjectDetectionsResults/found object3.jpg', image3)
cv2.imwrite('output/ObjectDetectionsResults/found object4.jpg', image4)
cv2.imwrite('output/ObjectDetectionsResults/found object5.jpg', image5)

#cv2.imwrite('output/ObjectDetectionsResults/kps.jpg', img2)