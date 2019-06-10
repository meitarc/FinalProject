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


'''
from objects import *

prototxtPath = 'MobileNetSSD_deploy.prototxt.txt'
caffemodelPath = 'MobileNetSSD_deploy.caffemodel'

arrayimg=readImagesToMakeCommonImage()
newSortedArrayimg=sortImageByFeachers(arrayimg)
i=0
range_list=findObject(newSortedArrayimg[0], prototxtPath, caffemodelPath)#list of ranges
'''
def firstFuncCheck(FirstImage):
    surf = cv2.xfeatures2d.SURF_create()
    img1 = np.array(FirstImage)
    kp, des = surf.detectAndCompute(img1, None)
    return kp,des
'''
kp,des = firstFuncCheck(newSortedArrayimg[0])

listOfObjects=[]
for i in range(range_list):
    descriptorOfObjects = []
    keyOfObjects = []
    for j,k in zip(kp, des):
    #for cor in array_Kp_pt:
        if ((j.pt[0]>=range_list[i][2]) and (j.pt[0]<= range_list[i][3])):
            if ((j.pt[1]>=range_list[i][0]) and (j.pt[1]<= range_list[i][1])):
                keyOfObjects.append(j)
                descriptorOfObjects.append(k)
                object_tupel = (keyOfObjects,descriptorOfObjects)
    listOfObjects.append(object_tupel)
'''
def IntersectOfImages2(listOfObjects,newSortedArrayimg):
    i=0
    j=0
    listOfMatches = []
    for i in range (0,len(listOfObjects)):
        arraykp = listOfObjects[i][0]
        arraydes = listOfObjects[i][1]
        num = len(arraykp)
        for j in range (1,len(newSortedArrayimg)):
            x,arraykp,arraydes = funcCheck2(arraykp,arraydes,newSortedArrayimg[j])
        ratio = x/num
        matchesTupel = (arraykp,arraydes,ratio)
        listOfMatches.append(matchesTupel)
    return listOfMatches
#listOfMatches = IntersectOfImages2(listOfObjects,newSortedArrayimg)



'''
croped=newSortedArrayimg[0]
for i in range(0,len(listOfMatches)):
    if(listOfMatches[i][2]>0.8):
        croped=imageDeleteObject(croped,range_list[i])
newSortedArrayimg[0] = croped


kp_1,des_1 = IntersectOfImages(newSortedArrayimg)
dictionary = CreateDict(kp_1,des_1)
for i in listOfMatches:
    for j,k in zip(i[0],i[1]):
        dictionary.update({j.pt:(j,k)})
clusters = DB_SCAN(kp_1,25)
objectS_list = []
for i in listOfMatches:
    for j in i[0]:
        objectS_list.append(j.pt)
    clusters.append(objectS_list)
'''


def keyOfObject(range_list, kp, des):
    listOfObjects = []
    for i in range(0, len(range_list)):
        descriptorOfObjects = []
        keyOfObjects = []
        for j, k in zip(kp, des):
            # for cor in array_Kp_pt:
            if ((j.pt[0] >= range_list[i][2]) and (j.pt[0] <= range_list[i][3])):
                if ((j.pt[1] >= range_list[i][0]) and (j.pt[1] <= range_list[i][1])):
                    keyOfObjects.append(j)
                    descriptorOfObjects.append(k)
                    object_tupel = (keyOfObjects, descriptorOfObjects)
        listOfObjects.append(object_tupel)
    return listOfObjects


from matplotlib.pyplot import *
from scipy.spatial import Delaunay
from testofBoundery import function2
from scipy.spatial import Delaunay

def main(serverFolder,clientImg,outputFolder,threshold):
    import cv2
    getFolder(outputFolder)
    #imports:
    #from SURF2 import DB_SCAN
    #from align.align import *




    #server side:
    # getting big image array, splitting to smaller arrays
    # and then, for each array do the following
    #function getting array of images, returning the kps and des of the intersect of all images
    ###
    '''
    from matplotlib import pyplot as plt
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = np.array(img2)
    kp, des = sift.detectAndCompute(img1, None)
    for i in kp:
        #print(type(i))
        cv2.circle(img1, (int(i.pt[0]), int(i.pt[1])), 10, (255, 0, 255), -1)
    plt.imshow(img1,),plt.show()
    print("show clusterts to boris:")
    clusters = DB_SCAN(kp,100)
    print("finish showing clusterts to boris")
    '''
    ###
    #MAIN
    #threshold=threshold # precentege of matches in order to consider good cluster
    import os
    arrayServerImgs=[]
    folderPath=serverFolder
    for filename in os.listdir(folderPath):
        arrayServerImgs.append(folderPath + "/" + filename)




    prototxtPath = 'MobileNetSSD_deploy.prototxt.txt'
    caffemodelPath = 'MobileNetSSD_deploy.caffemodel'

    i = 0
    arrayimg=readImagesToMakeCommonImage(arrayServerImgs)
    newSortedArrayimg=sortImageByFeachers(arrayimg) # sort images by number of features:
    yoloLabels = 'yoloLabels.txt'
    yoloWeights = 'yolov3.weights'
    yoloConfig = 'yolov3.cfg'
    threshold_ob = 0.5

    range_list = findObjectsUsingYOLO(newSortedArrayimg[0],yoloLabels,yoloWeights,yoloConfig,threshold_ob)
    print(len(range_list), "len range list")
    #range_list = findObject(newSortedArrayimg[0], prototxtPath, caffemodelPath)  # list of ranges
    kp, des = firstFuncCheck(newSortedArrayimg[0])
    listOfObjects=keyOfObject(range_list,kp,des)
    print(len(listOfObjects),"list of object")
    listOfMatches = IntersectOfImages2(listOfObjects, newSortedArrayimg)
    print(len(listOfMatches),"list of matches")
    print(listOfMatches)
    croped = newSortedArrayimg[0]

    new_listOfMatches=[]
    listOfNumbers=[]
    for i in range(0, len(listOfMatches)):
        #print(listOfMatches[i][2])
        if (listOfMatches[i][2] > 0.4):
            croped = imageDeleteObject(croped, range_list[i])
            cv2.imwrite(outputFolder + '/croppedBoris'+str(i)+'.jpg', croped)
            listOfNumbers.append(i)
            t_list = (listOfMatches[i][0],listOfMatches[i][1])
            new_listOfMatches.append(t_list)
    newSortedArrayimg[0] = croped

    cv2.imwrite(outputFolder + '/croppedBoris.jpg', newSortedArrayimg[0])
    print("go check boris")
    kp_1, des_1 = IntersectOfImages(newSortedArrayimg)# find inersect of features on all images:
    dictionary = CreateDict(kp_1, des_1) #dictionary between coordinates and keypoints+descriptors:

    for i in new_listOfMatches:
        for j, k in zip(i[0], i[1]):
            dictionary.update({j.pt: (j, k)})

    clusters = DB_SCAN(kp_1, 10) #clustering the kp according to coords
    NClustersWObjects = len(clusters)
    objectS_list = []
    for i in new_listOfMatches:
        for j in i[0]:
            objectS_list.append(j.pt)
        clusters.append(objectS_list)


    # low value mean more clusters, 10-100 most likely, now we are on 20.
    dict=makeDictforOriginalClusters(clusters)
    print("Number of original clusters: ",len(clusters))
    #given GPS send cluster to client
    #client side:
    #client image
    #image=cv2.imread("source/14b.jpg")

    #experiment1:
    #image=cv2.imread("source/115.jpg")

    #experiment2:
    #image=cv2.imread("source/Experiment2/204.jpg")

    #experiment3:
    ##new row:
    #imReg, h = alignImages(image,newSortedArrayimg[len(newSortedArrayimg)-1])
    #for each cluster, if found in camera image, take it off from cameras image:
    image=cv2.imread(clientImg) # read client image
    arrayOfGoodclusters,flagsOfGoodClusters,arrayOfBadclusters,flagsOfBadClusters,newListOfNumbers,count_originals = makegoodclusters(clusters,dictionary,image,threshold,NClustersWObjects,listOfNumbers) #find good clusters and bad clusters

    dict2=makeDictforGoodClusters(arrayOfGoodclusters,flagsOfGoodClusters)
    dict3=makeDictforBadClusters(arrayOfBadclusters,flagsOfBadClusters)
    #croppedimage=function2(arrayOfGoodclusters)

    #croppedimage= croppedmatchingareas(image,arrayOfGoodclusters)
    #img=cv2.imread("115.jpg")
    # Constructing the input point data


    #arrayOfGoodclusters=makegoodclusters(clusters,dictionary,image,threshold)

    # drop the areas of clusters found in the client image that match the server image
    croppedimage = makecroppedimage(arrayOfGoodclusters,image,newListOfNumbers,count_originals,range_list) #crop good clusters from client image
    cv2.imwrite(outputFolder+'/cropped2.jpg', croppedimage)
    #returnCroppedParts(croppedimage,newSortedArrayimg[len(newSortedArrayimg)-1],dict2,dict)
    print("CROPPED ! GO CHECK IT OUT !")


    Newclusters,Newdictionary,kp2,des2 = clustersOfCroppedImage(croppedimage) # sift and cluster kp's on client image after crop
    secondRange_list = findObjectsUsingYOLO(croppedimage, yoloLabels, yoloWeights, yoloConfig, threshold_ob)
    newListOfObjects = keyOfObject(secondRange_list, kp2, des2)
    new_listOfNumbers = []
    new_cropped=croppedimage
    #new_cropped = imageDeleteObject(croppedimage, secondRange_list[0])
    #new_listOfNumbers.append(0)
    for i in range(0,len(secondRange_list)):
        new_cropped = imageDeleteObject(new_cropped, secondRange_list[i])
        new_listOfNumbers.append(i)

    #Newclusters2, Newdictionary2, kp3, des3 = clustersOfCroppedImage(new_croped)
    Newclusters2, Newdictionary2, kp3, des3 = clustersOfCroppedImage(new_cropped)

    #take out the new clusters in order to send
    for i in newListOfObjects:
        for j, k in zip(i[0], i[1]):
            Newdictionary2.update({j.pt: (j, k)})

    NClustersWObjects2 = len(Newclusters2)
    NobjectS_list = []
    for i in new_listOfMatches:
        for j in i[0]:
            NobjectS_list.append(j.pt)
        Newclusters2.append(objectS_list)

    #######################################################################################
    #newimage=makecroppedimageseconduse(Newclusters,croppedimage)
    newimage=makecroppedimage(Newclusters,new_cropped,new_listOfNumbers,NClustersWObjects2,secondRange_list) # newimage is the cropped image after cropping sift clusters from it
    #newimage=croppedmatchingareas(croppedimage,Newclusters)
    cv2.imwrite(outputFolder+'/clusters_of_cropped2.jpg', newimage)
    imagetosend=croppedimage-newimage  # the negetivity in order to send to. makes it that we send just the clusters we found after first cropped
    cv2.imwrite(outputFolder+'/clusters_to_send2.jpg', imagetosend)

    #for better understanding of image, on server side, return parts of good clusters and bad clsuters:
    imagetotakeclustersfrom = newSortedArrayimg[len(newSortedArrayimg)-1]
    imgafterGoodclustersreturn = returnCroppedParts(imagetosend,imagetotakeclustersfrom,dict2,dict)

    #imgafterBadclustersreturn = returnCroppedParts2(imgafterGoodclustersreturn,imagetotakeclustersfrom,dict3, dict)

    #func3()

    #dim=godel
    #image= cv2.resize(imagetotakeclustersfrom[curtainspot],dim,interpolation=cv2.INTER_AREA)
    #imagetosend[x,y of smola lemata]=image
    #we need to take clusters from "imagetotakeclustersfrom" in places of dict()

    #cv2.imwrite('project.jpg',l_img)
    '''
    '''

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
    import cv2
    import numpy as np

threshhold=0.25
main("source/3.6.19/2/server","source/3.6.19/2/client/186.jpg","source/3.6.19/2/output/"+str(threshhold),threshhold)


#threshhold2=0.5
#for j in range(0,11):
#	threshhold=(j/10)
#	threshhold2=threshhold+0.05
#	threshhold2=float("%.2f" % threshhold2)
#    main("source/3.6.19/1/server","source/3.6.19/1/client/97.jpg","source/3.6.19/1/output/"+str(threshhold),threshhold)
#    main("source/3.6.19/6/server", "source/3.6.19/6/client/19.jpg", "source/3.6.19/6/output/"+str(threshhold2), threshhold2)
#

