import cv2
import numpy as np


######## YOLO CODE ##########

'''
    example input:
    imagePath = 'images/208.jpg'
    imageFile = cv2.imread(imagePath)
    threshold = 0.6
    yoloConfig = 'yolov3.cfg'
    yoloWeights = 'yolov3.weights'
    yoloLabels = 'yoloLabels.txt'
    scale = 0.00392
'''

    # function to get the output layer names
    # in the architecture
def get_output_layers(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def findObjectsUsingYOLO(imagePath, yoloLabels, yoloWeights, yoloConfig, threshold):
    return findObjectsUsingYOLOWithScale(imagePath, yoloLabels, yoloWeights, yoloConfig, 0.00392, threshold)

def findObjectsUsingYOLOWithScale(imagePath, yoloLabels, yoloWeights, yoloConfig, scale, threshold):
    #image = cv2.imread(imagePath)
    image = imagePath
    Width = image.shape[1]
    Height = image.shape[0]

    # read class names from text file
    classes = None
    with open(yoloLabels, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    net = cv2.dnn.readNet(yoloWeights, yoloConfig)
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    detectionList = []

    # for each detetion from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2

                startY = (int)(y)
                endY = (int)(y + h)
                startX = (int)(x)
                endX = (int)(x + w)
                range = (startY, endY, startX, endX)
                detectionList.append(range)
    return detectionList


######## YOLO END ##########

######### returnSubImage( image, startY, endY, startX, endX) #########
def returnSubImage( image, startY, endY, startX, endX):
    temp = image.copy()
    temp = temp[startY:endY, startX:endX]
    return temp
'''
#MAIN TEST
from functions import funcCheck1, funcCheck2

prototxtPath = 'ObjectDalgorithm/MobileNetSSD_deploy.prototxt.txt'
caffemodelPath = 'ObjectDalgorithm/MobileNetSSD_deploy.caffemodel'
#opening Image Example
image1 = cv2.imread('source/old_pics/115.jpg')
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
#cv2.imwrite('output/ObjectDetectionsResults/kps.jpg', img2)'''
