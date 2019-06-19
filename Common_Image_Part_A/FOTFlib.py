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
    #Object detection section: returns ranges(yStart,yEnd,xStart,xEnd) of each object
    range_list = findObjectsUsingYOLO(SortedArrayimg[0],yoloLabels,yoloWeights,yoloConfig,threshold_ob)

    flagarrayOfObjects=[]
    for i in range_list:
        flagarrayOfObjects.append(0)

    print("FLAGS ARE ",flagarrayOfObjects)
    print(len(range_list),"the len of all objects")
    #get the key,des of the first picture with the most features
    kp, des = firstFuncCheck(SortedArrayimg[0])
    #list of tupels(key,des) of each object
    listOfObjects = keyOfObject(range_list,kp,des)
    print(len(listOfObjects),"should be 7")
    #list of matched objects from all the pictures
    listOfMatches = IntersectOfImages2(listOfObjects, SortedArrayimg)
    #print(len(range_list), "len range list, number of objects")
    #print(len(listOfObjects), "list of object")
    #print(len(listOfMatches),"list of matches")

    croped = SortedArrayimg[0]
    #remove the best matched objects with high ration from the first picture
    croped, new_listOfMatches, listOfNumbers,flagarrayOfObjects = matchedObjects(listOfMatches, range_list, croped,flagarrayOfObjects)
    print(len(new_listOfMatches),"list of objects match in all pictures")
    print(len(listOfNumbers), "list of objects match in all pictures")
    serverimg = SortedArrayimg[0]

    counter=0
    print("FLAGS ARE ", flagarrayOfObjects)
    newArray=[]
    newIndexarray=[]
    for i in flagarrayOfObjects:
        if i==1:
            print(i)
            print(range_list[counter])
            newArray.append(counter)
            newIndexarray.append(0)
            serverimg = imageDeleteObject(serverimg, range_list[counter])
            cv2.imwrite(outputFolder + "/croppedOmriServer" + str(counter) + ".jpg", serverimg)
            flagarrayOfObjects[counter]=0 #initialize the array
        else:
            print("NOTTTT ",i)
        counter = counter + 1
    print(newArray)
    print(newIndexarray)
    #overwrite the first picture with the croped
    SortedArrayimg[0] = croped
    cv2.imwrite(outputFolder+'/after_objects_off.jpg', croped)
    kp_1, des_1 = IntersectOfImages(SortedArrayimg)# find inersect of features on all images:
    dictionary = CreateDict(kp_1, des_1) #dictionary between coordinates and keypoints+descriptors:
    #adding to the current dictionary the clusters of objects
    dictionary=updateDict(dictionary,new_listOfMatches)
    print(len(dictionary))
    #build clusters and then we adding the objects clusters
    clusters,NClustersWObjects=updateCluster(kp_1,dbscan_epsilon,new_listOfMatches)
    print(NClustersWObjects," number of cluster without objects")
    print(len(clusters),"all the clusters")

    dict=makeDictforOriginalClusters(clusters)
    print("Number of original clusters: ",len(clusters))



    #CLIENT
    ClientImage=cv2.imread(clientImg) # read client image




    ClientImage=cv2.imread(clientImg) # read client image
    #good clusters-matched clusters between the server and client

    arrayOfGoodclusters,flagsOfGoodClusters,arrayOfBadclusters,flagsOfBadClusters,newListOfNumbers,count_originals,newIndexarray = makegoodclusters(clusters,dictionary,ClientImage,threshold,NClustersWObjects,listOfNumbers,newIndexarray) #find good clusters and bad clusters
    print("new index array after change ",newIndexarray)
    print(count_originals,"the number of regular clusers that are good")
    print(len(newListOfNumbers),"the number of good objects")

    dict2=makeDictforGoodClusters(arrayOfGoodclusters[:count_originals],flagsOfGoodClusters)
    #dict3=makeDictforBadClusters(arrayOfBadclusters,flagsOfBadClusters)
    print(len(arrayOfGoodclusters),"array of good clusters")
    #drop the good clusters from the client image
    croppedimage = makecroppedimage(arrayOfGoodclusters,ClientImage,newListOfNumbers,count_originals,range_list) # drop the areas of clusters found in the client image that match the server image
    cv2.imwrite(outputFolder+'/cropped2.jpg', croppedimage)
    print("CROPPED ! GO CHECK IT OUT !")

    print("SECOND PLOT")
    #Object detection section: returns ranges(yStart,yEnd,xStart,xEnd) of each object-on the croppedimage
    secondRange_list = findObjectsUsingYOLO(croppedimage, yoloLabels, yoloWeights, yoloConfig, threshold_ob)
    print(len(secondRange_list),"objectssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss")

    new_cropped=croppedimage
    #remove the clients objects from the cropped image
    #for i in range(0,len(secondRange_list)):
    #    new_cropped = imageDeleteObject(new_cropped, secondRange_list[i])
    for j in secondRange_list:
        new_cropped = imageDeleteObject(new_cropped, j)


    print("third PLOT")
    #take out the new clusters from the client image in order to send
    Newclusters2, Newdictionary2, kp3, des3 = clustersOfCroppedImage(new_cropped,dbscan_epsilon)
    #newimage is the cropped image after cropping sift clusters from it
    newimage=makecroppedimage2(Newclusters2,new_cropped)
    cv2.imwrite(outputFolder+'/clusters_of_cropped2.jpg', newimage)
    cv2.imwrite(outputFolder+'/clusters_to_send2.jpg', croppedimage-newimage) # the negetivity in order to send to. makes it that we send just the clusters we found after first cropped
    #return the changed parts
    imagetosend =croppedimage-newimage
    imagetotakeclustersfrom = SortedArrayimg[len(SortedArrayimg)-1]
    imagetosend=returnCroppedParts(imagetosend,imagetotakeclustersfrom,dict2,dict) #for better understanding of image, on server side, return parts of good clusters and bad clsuters:
    returnobjects(imagetosend,SortedArrayimg[0],newIndexarray,range_list,outputFolder,newArray)

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
dbscan=10
#main("source/pics_for_tests/1/server", "source/pics_for_tests/1/client/6.jpg", "source/pics_for_tests/1/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/1/server", "source/pics_for_tests/1/client/10.jpg", "source/pics_for_tests/1/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/1/server", "source/pics_for_tests/1/client/13.jpg", "source/pics_for_tests/1/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/3/server", "source/pics_for_tests/3/client/19.jpg", "source/pics_for_tests/3/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/3/server", "source/pics_for_tests/3/client/90.jpg", "source/pics_for_tests/3/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/3/server", "source/pics_for_tests/3/client/277.jpg", "source/pics_for_tests/3/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/4/server", "source/pics_for_tests/4/client/115.jpg", "source/pics_for_tests/4/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/4/server", "source/pics_for_tests/4/client/121.jpg", "source/pics_for_tests/4/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/4/server", "source/pics_for_tests/4/client/183.jpg", "source/pics_for_tests/4/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/5/server", "source/pics_for_tests/5/client/115.jpg", "source/pics_for_tests/5/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/6/server", "source/pics_for_tests/6/client/97.jpg", "source/pics_for_tests/6/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/7/server", "source/pics_for_tests/7/client/186.jpg", "source/pics_for_tests/7/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/8/server", "source/pics_for_tests/8/client/120.jpg", "source/pics_for_tests/8/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/9/server", "source/pics_for_tests/9/client/73.jpg", "source/pics_for_tests/9/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/9/server", "source/pics_for_tests/9/client/77.jpg", "source/pics_for_tests/9/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/10/server", "source/pics_for_tests/10/client/115.jpg", "source/pics_for_tests/10/output/" + str(threshhold), threshhold,dbscan)
main("source/pics_for_tests/11/server", "source/pics_for_tests/11/client/204.jpg", "source/pics_for_tests/11/output/" + str(threshhold), threshhold,10)
#main("source/pics_for_tests/12/server", "source/pics_for_tests/12/client/87.jpg", "source/pics_for_tests/12/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/12/server", "source/pics_for_tests/12/client/88.jpg", "source/pics_for_tests/12/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/12/server", "source/pics_for_tests/12/client/113.jpg", "source/pics_for_tests/12/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/12/server", "source/pics_for_tests/12/client/115.jpg", "source/pics_for_tests/12/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/12/server", "source/pics_for_tests/12/client/117.jpg", "source/pics_for_tests/12/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/12/server", "source/pics_for_tests/12/client/122.jpg", "source/pics_for_tests/12/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/12/server", "source/pics_for_tests/12/client/127.jpg", "source/pics_for_tests/12/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/13/server", "source/pics_for_tests/13/client/20190602_172935(0).jpg", "source/pics_for_tests/13/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/14/server", "source/pics_for_tests/14/client/4.jpg", "source/pics_for_tests/14/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/14/server", "source/pics_for_tests/14/client/20190602_172914.jpg", "source/pics_for_tests/14/output/" + str(threshhold), threshhold,dbscan)
#main("source/pics_for_tests/15/server", "source/pics_for_tests/15/client/11.jpg", "source/pics_for_tests/15/output/" + str(threshhold), threshhold,dbscan)

