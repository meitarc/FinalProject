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
from functions import *
#MAIN
threshold=0.25
#server side:
# getting big image array, splitting to smaller arrays
# and then, for each array do the following
#function getting array of images, returning the kps and des of the intersect of all images
#img1=cv2.imread("1.jpg")
#img2=cv2.imread("2.jpg")
#img3=cv2.imread("3.jpg")
img4=cv2.imread("100.jpg")
img5=cv2.imread("101.jpg")
img6=cv2.imread("102.jpg")
#img7=cv2.imread("103.jpg")
#img8=cv2.imread("104.jpg")
#img9=cv2.imread("9.jpg")
#img11=cv2.imread("11.jpg")
#img12=cv2.imread("12.jpg")
#img13=cv2.imread("13.jpg")
#img14=cv2.imread("14.jpg")
#img15=cv2.imread("15.jpg")
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

arrayimg=[img4,img5,img6]
kp,des = IntersectOfImages(arrayimg)
dictionary = CreateDict(kp,des)
clusters = DB_SCAN(kp,20)
print("Number of original clusters: ",len(clusters))
#given GPS send cluster to client
#client side:
image=cv2.imread("115.jpg")
#for each cluster, if found in camera image, take it off from cameras image

arrayOfGoodclusters=makegoodclusters(clusters,dictionary,image,threshold)
print("Number of good clusters: ",len(arrayOfGoodclusters))

croppedimage=makecroppedimage(arrayOfGoodclusters,image)
cv2.imwrite('cropped.jpg', croppedimage)
Newclusters,Newdictionary = clustersOfCroppedImage(croppedimage)

#take out the new clusters in order to send
#
newimage=makecroppedimage(Newclusters,croppedimage)
cv2.imwrite('clusters_of_cropped.jpg', newimage)

cv2.imwrite('clusters_to_send.jpg', croppedimage-newimage)
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