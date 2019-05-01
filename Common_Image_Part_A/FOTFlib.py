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
from SURF2 import DB_SCAN
from functions import *
#MAIN
threshold=0.05
#server side:
# getting big image array, splitting to smaller arrays
# and then, for each array do the following
#function getting array of images, returning the kps and des of the intersect of all images
arrayimg=[]
kp,des = IntersectOfImages(arrayimg)
dictionary = CreateDict(kp,des)
clusters = DB_SCAN(kp,des)
#given GPS send cluster to client
#client side:
image=CameraImage
#for each cluster, if found in camera image, take it off from cameras image
arrayOfGoodclusters=[]
for cluster in clusters:
    if checkCluster(cluster,dictionary,image)>threshold:
        arrayOfGoodclusters.append(cluster)
sizes=[]
for cluster in clusters:
    minY, maxY, minX, maxX = corMinMax(cluster)
    SizeCenter = SizeandCenter(minY, maxY, minX, maxX)
    sizes.append(SizeCenter)

croppedimage= imageDeleteParts(image,sizes)

Newclusters,Newdictionary = clustersOfCroppedImage(croppedimage)


#in new cameras image(after parts removed) do funccheck
#cluster the featuers that returned.
#crop the parts of clusters found in cameras image and return them.
#to do - check for location of cropped parts in order to tell the server where they are located in server image.


