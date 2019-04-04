import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import cv2


def Save_Descripto(image1):
    path = 'dAndk.csv'
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = np.array(image1)
    kp1, des1 = sift.detectAndCompute(img1, None)
    df = pd.DataFrame(des1)
    df.to_csv(path)


def Save_Keypoint(imag1):
    path = 'dAndk.csv'
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = np.array(imag1)
    kp1, des1 = sift.detectAndCompute(img1, None)
    print(kp1[0])
    print(kp1[0].pt, kp1[0].size, kp1[0].angle, kp1[0].response, kp1[0].octave, kp1[0].class_id)
    key_dictonary = {}
    for key in kp1:
        key_pt = key.pt
        key_dictonary[key] = key_pt[0], key_pt[1], key.size, key.angle, key.response, key.octave, key.class_id
    df = pd.DataFrame(key_dictonary)
    df = df.transpose()
    df.to_csv('keysDF.csv')


# KeyPoint (float x, float y, float _size, float _angle=-1, float _response=0, int _octave=0, int _class_id=-1)
def Load_Keypoint(path):
    df = pd.read_csv(path)
    df = df.values
    _keypoints_array = []
    for item in df:
        _keypoints_array.append(item[0:])
    # print(_keypoints_array[0])
    keypointArray = []
    for key in _keypoints_array:
        thekey = cv2.KeyPoint(x=key[1], y=key[2], _size=key[3], _angle=key[4], _response=key[5], _octave=int(key[6]),
                              _class_id=int(key[7]))
        keypointArray.append(thekey)
    #print(keypointArray[0])
    #print(keypointArray[0].pt, keypointArray[0].size, keypointArray[0].angle, keypointArray[0].response, keypointArray[0].octave, keypointArray[0].class_id)


def load_Descripto(path):
    df = pd.read_csv(path)
    df = df.values
    _descripto_array = []
    # print(df)
    for desc in df:
        _descripto_array.append(desc)
    print(_descripto_array[0])


def funcCheck(image1, image2):
    # Initiate SIFT detector
    # print("start func Check")
    sift = cv2.xfeatures2d.SIFT_create()
    # changing from PIL to nparray to work with "detectandCompute"
    # find the keypoints and descriptors with SIFT
    img1 = np.array(image1)
    img2 = np.array(image2)
    # print("after converting to np")

    kp1, des1 = sift.detectAndCompute(img1, None)

    print(type(kp1))
    print(kp1[0])
    print(type(des1))
    print(des1[0])

    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # print("after flann")

    matches = flann.knnMatch(des1, des2, k=2)
    # print("after matches")
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # print("after matchesMASK")

    # ratio test as per Lowe's paper
    p = 0  # counter
    for i, (m, n) in enumerate(matches):
        # print("not matched")
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            ## Notice: How to get the index
            p = p + 1
            # print("matched")
            # print(p)
            pt1 = kp1[m.queryIdx].pt
            # print(pt1)
            pt2 = kp2[m.trainIdx].pt
            ## Draw pairs in purple, to make sure the result is ok
            cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 10, (255, 0, 255), -1)
            cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 10, (255, 0, 255), -1)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    # plt.imshow(img3,),plt.show()
    return p


img1 = cv2.imread('try0.jpg', 0)  # queryImage
# img2 = cv2.imread('try1.jpg', 0)

Save_Descripto(img1)
Save_Keypoint(img1)
load_Descripto('dAndk.csv')
Load_Keypoint('keysDF.csv')

# print(x)

from PIL import Image


def FetureCount(image1):
    images = np.array(image1)
    img1 = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    return len(kp1)
