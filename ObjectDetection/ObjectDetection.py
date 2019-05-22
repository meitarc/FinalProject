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
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe(prototxtPath, caffemodelPath)
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    # (note: normalization is done via the authors of the MobileNet SSD implementation)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()
    detectionList = list()
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > threshold:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            temp = (startY, endY, startX, endX)
            detectionList.append(temp)

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