#Comments:
'''
    1. Description of the project
    2. imports
    3. main functions
        Phase A, creating common image
        3.1.1 getting an array of images and splitting it to diffrent arrays according to image similarity
        3.1.2 for each array, find the intersect of keypoints
        3.1.3 cluster the keypoints using dbscan and save
        Phase B
        3.2.1  when given GPS, send the clusters and it's location in the image to the client
        Phase C - client side
        3.3.1 given client GPS, send to server ask for image(clsuters)
        3.3.2 check each cluster if found in camera's image
        3.3.3.1 if cluster found
        3.3.3.2 if cluster not found
        3.3.3.3 if area with no features on either side
        3.3.3 - to be decided
        Phase D - Server Side
        3.4.1 gets the non-matching parts.
        3.4.2 joining the clusters and new parts into an image.
    4. helpful functions
'''
from functions import *
outputFolder=''
from objects import *
import os
import cv2

#creating paths and basic information for the YOLO object detection algorithm.
yoloLabels = 'ObjectDalgorithm/yoloLabels.txt'
yoloWeights = 'ObjectDalgorithm/yolov3.weights'
yoloConfig = 'ObjectDalgorithm/yolov3.cfg'
threshold_ob = 0.5

def main(serverFolder,clientImg,outputFolder,threshold,dbscan_epsilon):
    # threshold - precentege of matches in order to consider good cluster

    getFolder(outputFolder)

    #server side:
    # getting big image array, splitting to smaller arrays
    # and then, for each array do the following
    #function getting array of images, returning the kps and des of the intersect of all images

    arrayServerImgs=[]
    folderPath=serverFolder
    for filename in os.listdir(folderPath):
        arrayServerImgs.append(folderPath + "/" + filename)

    #
    arrayimg=readImagesToMakeCommonImage(arrayServerImgs)
    newSortedArrayimg=sortImageByFeachers(arrayimg) # sort images by number of features:

    #If we have an array with many pictures, we divide the pictures to many, different arrays of images.
    allarray=divideArrayOfIMG(newSortedArrayimg, threshold2=0.01)

    newSortedArrayimg=allarray[0]

    range_list = findObjectsUsingYOLO(newSortedArrayimg[0],yoloLabels,yoloWeights,yoloConfig,threshold_ob)
    print(len(range_list), "len range list")

    kp, des = firstFuncCheck(newSortedArrayimg[0])
    listOfObjects=keyOfObject(range_list,kp,des)
    print(len(listOfObjects),"list of object")
    listOfMatches = IntersectOfImages2(listOfObjects, newSortedArrayimg)
    print(len(listOfMatches),"list of matches")

    croped = newSortedArrayimg[0]
    croped, new_listOfMatches, listOfNumbers = matchedObjects(listOfMatches, range_list, croped)
    newSortedArrayimg[0] = croped
    kp_1, des_1 = IntersectOfImages(newSortedArrayimg)# find inersect of features on all images:
    dictionary = CreateDict(kp_1, des_1) #dictionary between coordinates and keypoints+descriptors:

    dictionary=updateDict(dictionary,new_listOfMatches)

    clusters,NClustersWObjects=updateCluster(kp_1,dbscan_epsilon,new_listOfMatches)

    # low value mean more clusters, 10-100 most likely, now we are on 20.
    dict=makeDictforOriginalClusters(clusters)
    print("Number of original clusters: ",len(clusters))

    image=cv2.imread(clientImg) # read client image
    arrayOfGoodclusters,flagsOfGoodClusters,arrayOfBadclusters,flagsOfBadClusters,newListOfNumbers,count_originals = makegoodclusters(clusters,dictionary,image,threshold,NClustersWObjects,listOfNumbers) #find good clusters and bad clusters

    dict2=makeDictforGoodClusters(arrayOfGoodclusters,flagsOfGoodClusters)
    dict3=makeDictforBadClusters(arrayOfBadclusters,flagsOfBadClusters)

    # drop the areas of clusters found in the client image that match the server image
    croppedimage = makecroppedimage(arrayOfGoodclusters,image,newListOfNumbers,count_originals,range_list) #crop good clusters from client image
    cv2.imwrite(outputFolder+'/cropped2.jpg', croppedimage)
    print("CROPPED ! GO CHECK IT OUT !")


    Newclusters,Newdictionary,kp2,des2 = clustersOfCroppedImage(croppedimage,dbscan_epsilon) # sift and cluster kp's on client image after crop
    secondRange_list = findObjectsUsingYOLO(croppedimage, yoloLabels, yoloWeights, yoloConfig, threshold_ob)
    newListOfObjects = keyOfObject(secondRange_list, kp2, des2)
    new_listOfNumbers = []
    new_cropped=croppedimage

    for i in range(0,len(secondRange_list)):
        new_cropped = imageDeleteObject(new_cropped, secondRange_list[i])
        new_listOfNumbers.append(i)

    Newclusters2, Newdictionary2, kp3, des3 = clustersOfCroppedImage(new_cropped,dbscan_epsilon)

    #take out the new clusters in order to send
    Newdictionary2=updateDict(Newdictionary2,newListOfObjects)

    Newclusters2, NClustersWObjects2=updateCluster2(kp3,Newclusters2,new_listOfMatches)

    newimage=makecroppedimage(Newclusters2,new_cropped,new_listOfNumbers,NClustersWObjects2,secondRange_list) # newimage is the cropped image after cropping sift clusters from it
    cv2.imwrite(outputFolder+'/clusters_of_cropped2.jpg', newimage)

    imagetosend=croppedimage-newimage  # the negetivity in order to send to. makes it that we send just the clusters we found after first cropped
    cv2.imwrite(outputFolder+'/clusters_to_send2.jpg', imagetosend)

    #for better understanding of image, on server side, return parts of good clusters and bad clsuters:
    imagetotakeclustersfrom = newSortedArrayimg[len(newSortedArrayimg)-1]
    imgafterGoodclustersreturn = returnCroppedParts(imagetosend,imagetotakeclustersfrom,dict2,dict)

    #imgafterBadclustersreturn = returnCroppedParts2(imgafterGoodclustersreturn,imagetotakeclustersfrom,dict3, dict)

    '''
    src = cv2.imread('clusters_to_send.jpg', 1)
    
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    cv2.imwrite("test.jpeg", dst)
    
    counter=0
    for cluster in Newclusters:
        minY, maxY, minX, maxX = corMinMax(cluster)
        crop_img = croppedimage[int(minY):int(maxY),
                   int(minX):int(maxX)]
        cv2.imwrite('newcropped' + str(counter) + '.jpg', crop_img)
        counter=counter+1
    
    #in new cameras image(after parts removed) do funccheck
    #cluster the featuers that returned.
    #crop the parts of clusters found in cameras image and return them.
    #to do - check for location of cropped parts in order to tell the server where they are located in server image.
    '''

threshhold=0.25
main("source/3.6.19/4/server", "source/3.6.19/4/client/6.jpg", "source/3.6.19/4/output/" + str(threshhold), threshhold,10)

