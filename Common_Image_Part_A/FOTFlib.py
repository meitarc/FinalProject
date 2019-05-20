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

#imports:
#from SURF2 import DB_SCAN
from align.align import *
from functions import *

from matplotlib.pyplot import *
from scipy.spatial import Delaunay
from testofBoundery import function2
from scipy.spatial import Delaunay
#MAIN
threshold=0.25
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
#pic=cv2.imread("14b.jpg")
#tryit(pic)
arrayimg=readImagesToMakeCommonImage()
# sort images by number of features:
newSortedArrayimg=sortImageByFeachers(arrayimg)
#find inersect of features on all images:
kp,des = IntersectOfImages(newSortedArrayimg)
#dictionary between coordinates and keypoints+descriptors:
dictionary = CreateDict(kp,des)
#clustering the kp according to coords
# the value isL low for more cluster, 10-100 most likely, now we are on 20.
clusters = DB_SCAN(kp,20)
####

dict={}#dict of dictionary and flag, for first photo
for i in range(0,len(clusters)):
    minY, maxY, minX, maxX = corMinMax(clusters[i])
    #SizeCenter = SizeandCenter(minY, maxY, minX, maxX)
    SizeCenter = topleftAndSizes(minY, maxY, minX, maxX)
    dict.update({i:(SizeCenter)})
####
print("Number of original clusters: ",len(clusters))

#given GPS send cluster to client
#client side:
#client image
image=cv2.imread("source/115.jpg")
##new row:
imReg, h = alignImages(image,newSortedArrayimg[len(newSortedArrayimg)-1])
#for each cluster, if found in camera image, take it off from cameras image:
arrayOfGoodclusters,flagsOfGoodClusters = makegoodclusters(clusters,dictionary,imReg,threshold)
#arrayOfGoodclusters = makegoodclusters(clusters,dictionary,imReg,threshold)


dict2={}
print("flags")
#create dict2 for client image
for i in range (0,len(arrayOfGoodclusters)):
    minY, maxY, minX, maxX = corMinMax(arrayOfGoodclusters[i])
    #SizeCenter = SizeandCenter(minY, maxY, minX, maxX)
    SizeCenter=topleftAndSizes(minY, maxY, minX, maxX)
    z=flagsOfGoodClusters[i]
    dict2.update({z:SizeCenter})
    print(i,z)

#croppedimage=function2(arrayOfGoodclusters)

#croppedimage= croppedmatchingareas(image,arrayOfGoodclusters)
#img=cv2.imread("115.jpg")
# Constructing the input point data


#arrayOfGoodclusters=makegoodclusters(clusters,dictionary,image,threshold)
print("Number of good clusters: ",len(arrayOfGoodclusters))
# drop the areas of clusters found in the client image that match the server image
croppedimage=makecroppedimage(arrayOfGoodclusters,imReg)
#croppedimage=makecroppedimage(arrayOfGoodclusters,imReg)
cv2.imwrite('output/cropped2.jpg', croppedimage)
#returnCroppedParts(croppedimage,newSortedArrayimg[len(newSortedArrayimg)-1],dict2,dict)

print("CROPPED ! GO CHECK IT OUT !")
Newclusters,Newdictionary = clustersOfCroppedImage(croppedimage)

#take out the new clusters in order to send

#newimage=makecroppedimageseconduse(Newclusters,croppedimage)
newimage=makecroppedimage(Newclusters,croppedimage)

#newimage=croppedmatchingareas(croppedimage,Newclusters)
cv2.imwrite('output/clusters_of_cropped2.jpg', newimage)

cv2.imwrite('output/clusters_to_send2.jpg', croppedimage-newimage)

imagetosend=croppedimage-newimage

imagetotakeclustersfrom=newSortedArrayimg[len(newSortedArrayimg)-1]


returnCroppedParts(imagetosend,newSortedArrayimg[len(newSortedArrayimg)-1],dict2,dict)


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
