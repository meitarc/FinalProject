
import cv2
import numpy as np

from DBSCAN import function

def extractKeyPt(kp1):
    array_Kp_pt = []
    for kp in kp1:
        array_Kp_pt.append(kp.pt)
    return array_Kp_pt

def funcCheck1(image1, image2):
    print("my func")
    # Initiate SIFT detector
    #sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
    # changing from PIL to nparray to work with "detectandCompute"
    # find the keypoints and descriptors with SIFT
    img1 = np.array(image1)
    img2 = np.array(image2)
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)
    #print(kp1)
    #print()
    #print(des1)
    #print()
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # http://answers.opencv.org/question/35327/opencv-and-python-problems-with-knnmatch-arguments/
    matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    p = 0  # counter
    #print(matches)
    #arr=[]
    okp=[]
    odes=[]

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            #print(i,m,n)
            #print(m.trainIdx)
            matchesMask[i] = [1, 0]
            ## Notice: How to get the index
            p = p + 1
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt

            #o1=(kp1[m.queryIdx],des1[m.queryIdx])
            okp.append(kp1[m.queryIdx])
            odes.append(des1[m.queryIdx])

            #tup=(o1,o2)
            #arr.append(tup)
            ## Draw pairs in purple, to make sure the result is ok
            cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 10, (255, 0, 255), -1)
            cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 10, (255, 0, 255), -1)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    # plt.imshow(img3,),plt.show()
    #print(okp)
    #print(odes)
    return p,okp,odes  # returns number of best matches,and all keypoints of first img



def funcCheck2(kp,des, image2):
    print("my func")
    # Initiate SIFT detector
    #surf = cv2.SURF(400)
    surf = cv2.xfeatures2d.SURF_create()
    #sift = cv2.xfeatures2d.SIFT_create()
    # changing from PIL to nparray to work with "detectandCompute"
    # find the keypoints and descriptors with SIFT
    #img1 = np.array(image1)
    img2 = np.array(image2)
    kp1, des1 = kp,des

    kp2, des2 = surf.detectAndCompute(img2, None)
    #print(kp1)
    #print()
    #print(des1)
    #print()
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # http://answers.opencv.org/question/35327/opencv-and-python-problems-with-knnmatch-arguments/
    matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    p = 0  # counter
    #print(matches)
    #arr=[]
    okp=[]
    odes=[]
    okp2=[]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            #print(i,m,n)
            #print(m.trainIdx)
            matchesMask[i] = [1, 0]
            ## Notice: How to get the index
            p = p + 1
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            #o1=(kp1[m.queryIdx],des1[m.queryIdx])
            okp.append(kp2[m.trainIdx])
            okp2.append(kp1[m.queryIdx])
            odes.append(des2[m.trainIdx])
            #tup=(o1,o2)
            #arr.append(tup)
            ## Draw pairs in purple, to make sure the result is ok
            #cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 10, (255, 0, 255), -1)
            cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 10, (255, 0, 255), -1)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    # plt.imshow(img3,),plt.show()
    print(okp[0],okp2[0])
    return p,okp,odes  # returns number of best matches,and all keypoints of first img



imageONE = cv2.imread('try2.jpg')
imageTwo = cv2.imread('try0.jpg')
imageThree = cv2.imread('try1.jpg')
imageSix = cv2.imread('try2.jpg')
imageFour = cv2.imread('try0.jpg')
from matplotlib import pyplot as plt

arrayodPics=[imageONE,imageTwo,imageThree,imageFour,imageSix]
#1
x,arraykp,arraydes=funcCheck1(arrayodPics[0],arrayodPics[1])
print(len(arraykp))
image1=arrayodPics[0]
#sift = cv2.xfeatures2d.SIFT_create()
#surf = cv2.SURF(400)
surf = cv2.xfeatures2d.SURF_create()
# changing from PIL to nparray to work with "detectandCompute"
# find the keypoints and descriptors with SIFT
img1 = np.array(image1)
kp1, des1 = surf.detectAndCompute(img1, None)
print("kp1: ",len(kp1))
image3=arrayodPics[0]
img3= np.array(image3)

for i in kp1:
    # print(type(i))
    cv2.circle(img1, (int(i.pt[0]), int(i.pt[1])), 10, (255, 0, 255), -1)
plt.imshow(img1, ), plt.show()

for i in arraykp:
    # print(type(i))
    cv2.circle(img3, (int(i.pt[0]), int(i.pt[1])), 10, (255, 0, 255), -1)
plt.imshow(img3, ), plt.show()



#2
x,arraykp,arraydes=funcCheck2(arraykp,arraydes,arrayodPics[2])
print(len(arraykp))

x,arraykp,arraydes=funcCheck2(arraykp,arraydes,arrayodPics[3])
print(len(arraykp))
#3
x,arraykp,arraydes=funcCheck2(arraykp,arraydes,arrayodPics[4])

print(len(arraykp))

'''
for i in range(0,len(arrayodPics)):
    x,arraykp,arraydes=funcCheck2(arraykp,arraydes,arrayodPics[i])
    print(len(arraykp))
    from matplotlib import pyplot as plt
    image = arrayodPics[i]
    img = np.array(image)
    for i in arraykp:
        # print(type(i))
        cv2.circle(img, (int(i.pt[0]), int(i.pt[1])), 10, (255, 0, 255), -1)
    plt.imshow(img, ), plt.show()
'''
image = arrayodPics[4]
img = np.array(image)

arrayk=extractKeyPt(arraykp)
print("show function dbscan result")
function(arrayk)
for i in arrayk:
    # print(type(i))
    cv2.circle(img, (int(i.pt[0]), int(i.pt[1])), 10, (255, 0, 255), -1)
plt.imshow(img, ), plt.show()

#print(arraykp,arraydes)
'''


xy=funcCheck(imageONE,imageThree)
z=funcCheck(imageTwo,imageThree)
listx=[]
for i in x:
    listx.append(i[0])
    listx.append(i[1])
listy=[]
for i in y:
    listy.append(i[0])
    listy.append(i[1])
listz=[]
for i in z:
    listz.append(i[0])
    listz.append(i[1])
print(listx)
print(listy)
print(type(listx))
print()
u = set.intersection(set(listx),set(listy))
print(u)

#print(u)

#u = set.intersection(set(w1), set(w2), set(w3))
#print(u)

res1 = kp1[m.queryIdx].response
oct1 = kp1[m.queryIdx].octave
angle1 = kp1[m.queryIdx].angle
size1 = kp1[m.queryIdx].size
id1 = kp1[m.queryIdx].class_id

tup = (res1, oct1, angle1, size1, id1)
arr.append(tup)
'''