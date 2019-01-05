
### getFramesFromVideo ###

#Gets a video input path, an output path for the folder to contain the images and every given milliSeconds

#If the milliSeconds is 0, the default value is every possible frame

#videoInputPath example: 'D:\\project files\\information extraction\\mp4\\canon750D.mp4'
#outputFolderPath example: 'D:\\project files\\information extraction\\imgExtraction\\test'
#milliSeconds example: 1000 for 1 second, 0 for every frame

### getFramesFromVideo ###
#function gets:
    #a single video file input path (videoInputPath)
    #output folder for the images for a single video file (outputFolderPath) 
    #frames to convert to images in every milli-seconds
        #example: 1000 milli-seconds will take a single frame from a video from every 1 second mark
def getFramesFromVideo( videoInputPath, outputFolderPath, milliSeconds ):
    import cv2
    vidcap = cv2.VideoCapture(videoInputPath) 

    success,image = vidcap.read()
    count = 1
    success = True
    while success:
        cv2.imwrite((outputFolderPath + '\\' + 'frame_%d.jpg') % count, image)     
        if (milliSeconds > 0):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(milliSeconds*count))
        success,image = vidcap.read()
        #print 'Read a new frame: ', success
        count += 1
    #releasing the video file after we're done to free memmory space.
    vidcap.release()
   #return

### getFramesFromAllVideos ###

#this function does the same thing as getFramesFromVideo, but for all video files within all sub-folders
#the general protocol is:
    #gets a path for a folder
    #makes a list of all the sub-folders within said path's folder.
    #for each sub-folder, checks files withint it.
        #if the file is a video file, then creates a folder within the same folder with the same name as the original video.
        #within the new folder, the images extracted from the frames within the video are saved.
        #the sub-folder for the images is names as the video file.
def getFramesFromAllVideos(videoInputPath, outputFolderPath, milliSeconds):
    if (outputFolderPath == None):
        outputFolderPath = videoInputPath
    #saving all sub-folder' names
    subFolders = getAllSubFolderNames(videoInputPath)
    for folderName in subFolders:
        #ensuring the sub-file exists in the given path. if not; then creating the sub folder using the original video's filename. 
        if (isVideoFile(folderName)):
            ensureDir(videoInputPath + "\\" + getFileName(folderName) + "\\")
            ensureDir( outputFolderPath + "\\" + getFileName(folderName) + "\\")
            #after we made sure the folder for the images exists, we start capturing all the frames into images.
            getFramesFromVideo(videoInputPath + "\\" + folderName, outputFolderPath + "\\" + getFileName(folderName), milliSeconds)

### ensureDir ###

#Gets a path and ensures the folder exists.
#If the folder doesn't exist, then it's created.

#file_path example: 'D:\\folder\\newFolder\\'

def ensureDir(file_path):
    import os
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
### getAllSubFolderNames ###

#Gets a path, if the folder exists, then creates a list of all the sub-folder and file names and returns it.
#If the folder doesn't exist, then returns an empty list.

#folderPath example: 'D:\\project files\\information extraction\\mp4'

def getAllSubFolderNames(folderPath):
    import os, subprocess
    #the list of names
    folderNames = []
    
    if not (os.path.exists(folderPath)):
        return folderNames 
    
    for subFolderName in os.listdir(folderPath):
        folderNames.append(subFolderName)
        
    return folderNames


### getFileType ###
#getting the ending of the file, example: ".mp4"
def getFileType( fileName ):
    import os.path
    extension = os.path.splitext(fileName)[1]
    return extension
### getFileName ###
#getting the ending of the file, example: ".mp4"
def getFileName( fileName ):
    import os.path
    extension = os.path.splitext(fileName)[0]
    return extension

### getFramesFromAllVideos ###
#gets a string, if the string contains any known file formats ending (example: ".mp4") then returns a boolean of 'true', otherwise return 'false'
def isVideoFile( fileName ):
    if (getFileType(fileName).capitalize() == ".mp4"):
        return True
    else:
        return False

def isImageFile(fileName):
    if (getFileType(fileName).capitalize() == ".jpg"):
        return True
    else:
        return False
    
### getImageWidth ###
#Gets an img as defined by cv2.imread(filePath)
#Returns the Width of the given image
def getImageWidth( image ):
    return len(image.tolist()[0])

### getImageHeight ###
#Gets an img as defined by cv2.imread(filePath)
#Returns the Height of the given image
def getImageHeight( image ):
    return len(image.tolist())

### getImageResolution ###
#Gets an img as defined by cv2.imread(filePath)
#Returns a string of the image's Resolution, given the format: WidthXHeight
#Example for return: '1920X1080'
def getImageResolution( image ):
    return ("%d" % getImageWidth(image) + "X" + "%d" % getImageHeight(image))

### getPixelsList ###
#Gets an img as defined by cv2.imread(filePath)
#Returns a list of pixels as defined by an image.
#Example for the first pixel: getPixelsList( image )[0][0] = [19, 18, 17]
#Example for the first color of the first pixel(Red): getPixelsList( image )[0][0][0] == 19
#Example for the second color of the first pixel(Green): getPixelsList( image )[0][0][1] == 18
def getPixelsList( image ):
    return image.tolist()
    
################ Finding Points-Of-Intrests ##################### 

#for starters, we will convert our images to Black-And-White (no gray-scale)
 
### convertImageToBlackAndWhite ###
def convertImageToBlackAndWhiteByThreshold(image, Threshold):
    import cv2
    copy = image.copy()
    for pixel in copy:
        for rgb in pixel:
            if (rgb[0] + rgb[1] + rgb[2] > Threshold):
                #turning the pixel black
                rgb[0] = 0
                rgb[1] = 0
                rgb[2] = 0
            else:
                #turning the pixel white
                rgb[0] = 255
                rgb[1] = 255
                rgb[2] = 255
    return copy
    
### convertImageToBlackAndWhite ###
#Gets:
    #an image as defined by cv2.imread()
    #uses the convertImageToBlackAndWhiteByThreshold with a default Threshold of 120 
def convertImageToBlackAndWhite(image):
    return convertImageToBlackAndWhiteByThreshold(image, 120)
    
    
    
    

    
    
### Comparing 2 images and returning the similarities ###



### differenceBetweenTwoImages( image1, image2) ###
    #gets 2 images to compare.
    # returns: an image that represents the pixels that aren't equal   
    #The image must be normal RGB / GRB color schem
def differenceBetweenTwoImages( image1, image2):
    import cv2
    from skimage.measure import compare_ssim as ssim
    
    #need to check if the image is already color shemed to Gray-Scale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    (score, diff) = ssim(gray1, gray2, full=True, multichannel=True)
    diff = (diff * 255).astype("uint8")
    #need to turn the diff image to black and white, probably need to use the convertImageToBlackAndWhite function
    #diff2 = convertImageToBlackAndWhite(diff)
    #need to release images to free memmory
    return diff


#need to make sure that not the two-same comparessions are made more than once.
def saveAllDifferences(imagesFolderPath, differenceFolderPath):
    import cv2
    from skimage.measure import compare_ssim as ssim
    
    if (differenceFolderPath == None):
        differenceFolderPath = imagesFolderPath + "\\" + "differences"
    #saving all sub-folder' names
    subFolders = getAllSubFolderNames(imagesFolderPath)
    firstImg = subFolders[0]
    
    
    
    for folderName in subFolders:
        if (isImageFile(folderName) and isImageFile(firstImg) and firstImg != folderName):
            img1 = cv2.imread(imagesFolderPath + "\\"+ getFileName(folderName))
            diff = differenceBetweenTwoImages(img1, img2)
            cv2.imwrite(differenceFolderPath+"\\1.jpg", diff)


#as defined in compare.py

def mse(imageA, imageB):
    from skimage.measure import compare_ssim as ssim
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    #import imutils
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def compare_images(imageA, imageB, title):
    from skimage.measure import compare_ssim as ssim
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    #import imutils
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    return (m,s)

### similarityProbability(img1, img2) ###
def similarityProbability(img1, img2):
    from skimage.measure import compare_ssim as ssim
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    #import imutils
    return ssim(img1, img2)
    
    
### summingPixelCount( imageFolderPath ) ###
def summingPixelCount( imageFolderPath ):
    import cv2
    subFolders = getAllSubFolderNames(imageFolderPath)
    img1 = cv2.imread(imageFolderPath + "\\" + subFolders[0])
    i=0
    k=0
    
    array = []
    
    for i in range (0, getImageHeight(img1)):
            new = [] 
            for j in range (0, getImageWidth(img1)):
                new.append(0)
            array.append(new)
        
    
    for folder in subFolders:
        if (isImageFile(folder)):
            img2 = cv2.imread(imageFolderPath + "\\" + folder)
            tmpImage = differenceBetweenTwoImages(img1, img2)
            i = 0
            k = 0
            
            for pixels in tmpImage:
                for rgb in pixels:
                    array[i][k] += rgb[0]+rgb[1]+rgb[2]
                    k += 1
                i += 1
        img1 = img2
                
    return array
            

            
            
