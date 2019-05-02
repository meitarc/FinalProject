import cv2
import numpy as np
from SURF2 import DB_SCAN

def IntersectOfImages(arrayOfimages):
    x, arraykp, arraydes = funcCheck1(arrayOfimages[0], arrayOfimages[1])
    for i in range (2,len(arrayOfimages)):
        x,arraykp,arraydes = funcCheck2(arraykp,arraydes,arrayOfimages[i])
    return(arraykp,arraydes)

def CreateDict(kp,des):
    dict={}
    for i,j in zip(kp,des):
        key=(i.pt[0],i.pt[1])
        val=(i,j)
        item = {key:val}
        dict.update(item)
    return dict

def checkCluster(cluster,dictionary,image):
    arrayKP = []
    arrayDES = []
    for cor in cluster:

        values = dictionary.get(cor)

        if values is not None:
            key=values[0]
            val=values[1]
            arrayKP.append(key)
            arrayDES.append(val)
    p1, p2 = clientFuncCheck(arrayKP, arrayDES, image)
    return (p1 / len(p2))

def corMinMax(cluster):
    maxX = 0
    maxY = 0
    print(cluster)
    minX = cluster[0][0]
    minY = cluster[0][1]
    for cor in cluster:
            if cor[0] > maxX:
                maxX = cor[0]
            if cor[0] < minX:
                minX = cor[0]
            if cor[1] > maxY:
                maxY = cor[1]
            if cor[1] < minY:
                minY = cor[1]
    return minY,maxY,minX,maxX

def SizeandCenter(minY,maxY,minX,maxX):
    sizeX=maxX-minX
    sizeY=maxY-minY
    size=(sizeY,sizeX)
    center=(int(sizeY/2)+minY,int(sizeX)/2+minX)
    return center[0],center[1],int(size[1]/2),int(size[0]/2)

# imageDeleteParts
# Arguments:
#    Image - an image, as defined by cv2.imgread()
#    partsList - a list of tuples, that describe:
#                x, y, xradius, yradius values:
#                  x,y - location of the middle pixel.
#                  xradius, yradius - the radius around the x and y axis.
#The function returns the original image, after deleting the surrounding area given around the x,y.
def imageDeleteParts(Image, partsList):
    test = Image.copy()
    for range in partsList:
        print(range)
        test[int(range[1] - range[3]) : int(range[1] + range[3]), int(range[0] - range[2]) : int(range[0] + range[2]) ] = 0
    return test

def clustersOfCroppedImage(image1):
    #sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
    img1 = np.array(image1)
    kp, des = surf.detectAndCompute(img1, None)
    dictionary = CreateDict(kp, des)
    clusters = DB_SCAN(kp)
    return clusters,dictionary

def funcCheck1(image1, image2):
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
    return p,okp,odes

def clientFuncCheck(one, two, image2):
    print("my funcCheck 2:")
    # Initiate SIFT detector
    #sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
    # changing from PIL to nparray to work with "detectandCompute"
    # find the keypoints and descriptors with SIFT
    img2 = np.array(image2)

    kp1, des1 = one,two

    kp2, des2 = surf.detectAndCompute(img2, None)

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
    for i, (m, n) in enumerate(matches):

        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            ## Notice: How to get the index
            p = p + 1
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            ## Draw pairs in purple, to make sure the result is ok
            #cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 10, (255, 0, 255), -1)
            #cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 10, (255, 0, 255), -1)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    # plt.imshow(img3,),plt.show()
    return p, kp1  # returns number of best matches,and all keypoints of first img

