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
import cv2
import matplotlib.pyplot as plt
#creating paths and basic information for the YOLO object detection algorithm.
yoloLabels = 'ObjectDalgorithm/yoloLabels.txt'
yoloWeights = 'ObjectDalgorithm/yolov3.weights'
yoloConfig = 'ObjectDalgorithm/yolov3.cfg'
threshold_ob = 0.5

def main(serverFolder,clientImg,outputFolder,threshold,dbscan_epsilon):#threshold - precentege of matches in order to consider good cluster
    getFolder(outputFolder)

    #SERVER:
    #preprocessing, load, sort and divide images
    SortedArrayimg=buildaArrayImages(serverFolder)
    ###
    #Object detection section:
    range_list = findObjectsUsingYOLO(SortedArrayimg[0],yoloLabels,yoloWeights,yoloConfig,threshold_ob)
    print(len(range_list),"the len of all objects")
    kp, des = firstFuncCheck(SortedArrayimg[0])
    listOfObjects = keyOfObject(range_list,kp,des)
    print(len(listOfObjects),"should be 7")
    listOfMatches = IntersectOfImages2(listOfObjects, SortedArrayimg)
    #print(len(range_list), "len range list, number of objects")
    #print(len(listOfObjects), "list of object")
    #print(len(listOfMatches),"list of matches")
    croped = SortedArrayimg[0]
    croped, new_listOfMatches, listOfNumbers = matchedObjects(listOfMatches, range_list, croped)  #remove constant objects from first image
    print(len(new_listOfMatches),"list of objects match in all pictures")
    print(len(listOfNumbers), "list of objects match in all pictures")
    SortedArrayimg[0] = croped
    cv2.imwrite(outputFolder+'/after_objects_off.jpg', croped)
    kp_1, des_1 = IntersectOfImages(SortedArrayimg)# find inersect of features on all images:
    dictionary = CreateDict(kp_1, des_1) #dictionary between coordinates and keypoints+descriptors:
    dictionary=updateDict(dictionary,new_listOfMatches)
    print(len(dictionary))
    clusters,NClustersWObjects=updateCluster(kp_1,dbscan_epsilon,new_listOfMatches)
    print(NClustersWObjects," number of cluster without objects")
    print(len(clusters),"all the clusters")

    dict=makeDictforOriginalClusters(clusters)
    print("Number of original clusters: ",len(clusters))



    #CLIENT
    ClientImage=cv2.imread(clientImg) # read client image
    arrayOfGoodclusters,flagsOfGoodClusters,arrayOfBadclusters,flagsOfBadClusters,newListOfNumbers,count_originals = makegoodclusters(clusters,dictionary,ClientImage,threshold,NClustersWObjects,listOfNumbers) #find good clusters and bad clusters
    print(count_originals,"the number of regular clusers that are good")
    print(len(newListOfNumbers),"the number of good objects")
    dict2=makeDictforGoodClusters(arrayOfGoodclusters,flagsOfGoodClusters)
    #dict3=makeDictforBadClusters(arrayOfBadclusters,flagsOfBadClusters)
    print(len(arrayOfGoodclusters),"array of good clusters")
    croppedimage = makecroppedimage(arrayOfGoodclusters,ClientImage,newListOfNumbers,count_originals,range_list) # drop the areas of clusters found in the client image that match the server image
    cv2.imwrite(outputFolder+'/cropped2.jpg', croppedimage)
    print("CROPPED ! GO CHECK IT OUT !")

    print("SECOND PLOT")
    #Newclusters,Newdictionary,kp2,des2 = clustersOfCroppedImage(croppedimage,dbscan_epsilon) # sift and cluster kp's on client image after crop
    secondRange_list = findObjectsUsingYOLO(croppedimage, yoloLabels, yoloWeights, yoloConfig, threshold_ob)
    print(len(secondRange_list),"objectssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss")
    #newListOfObjects = keyOfObject(secondRange_list, kp2, des2)
    #new_listOfNumbers = []
    new_cropped=croppedimage
    for i in range(0,len(secondRange_list)):
        new_cropped = imageDeleteObject(new_cropped, secondRange_list[i])
        #new_listOfNumbers.append(i)
    print("third PLOT")
    Newclusters2, Newdictionary2, kp3, des3 = clustersOfCroppedImage(new_cropped,dbscan_epsilon)    #take out the new clusters in order to send
    #Newdictionary2=updateDict(Newdictionary2,newListOfObjects)
    #Newclusters2, NClustersWObjects2=updateCluster2(Newclusters2,new_listOfMatches)

    newimage=makecroppedimage2(Newclusters2,new_cropped) # newimage is the cropped image after cropping sift clusters from it
    cv2.imwrite(outputFolder+'/clusters_of_cropped2.jpg', newimage)
    cv2.imwrite(outputFolder+'/clusters_to_send2.jpg', croppedimage-newimage) # the negetivity in order to send to. makes it that we send just the clusters we found after first cropped
    imagetosend =croppedimage-newimage
    imagetotakeclustersfrom = SortedArrayimg[len(SortedArrayimg)-1]
    returnCroppedParts(imagetosend,imagetotakeclustersfrom,dict2,dict) #for better understanding of image, on server side, return parts of good clusters and bad clsuters:
    #imgafterBadclustersreturn = returnCroppedParts2(imgafterGoodclustersreturn,imagetotakeclustersfrom,dict3, dict)
    #
    reset()
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
main("source/pics_for_tests/5/server", "source/pics_for_tests/5/client/115.jpg", "source/pics_for_tests/5/output/" + str(threshhold), threshhold,10)