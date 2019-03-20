'''
#source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html
import cv2
from matplotlib import pyplot as plt
import numpy as np

#img = cv2.imread('fly.png',0)
img = cv2.imread('church.jpg',0)

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
#surf = cv2.SURF(400)
surf = cv2.xfeatures2d.SURF_create(400)

# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img,None)
print(len(kp))

###############################################
# Check present Hessian threshold
print(surf)


# We set it to some 50000. Remember, it is just for representing in picture.
# In actual cases, it is better to have a value 300-500
# surf.hessianThreshold = 50000

# Again compute keypoints and check its number.
kp, des = surf.detectAndCompute(img,None)

print(len(kp))

###################################################

img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)




plt.imshow(img2)
plt.show()


#######################################################
# Check upright flag, if it False, set it to True
#print(surf.upright)


#surf.upright = True

# Recompute the feature points and draw it
kp = surf.detect(img,None)
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)

plt.imshow(img2)
plt.show()

######################################################
# Find size of descriptor
print(surf.descriptorSize())

# That means flag, "extended" is False.
#surf.extended


# So we make it to True to get 128-dim descriptors.
#surf.extended = True
kp, des = surf.detectAndCompute(img,None)
print(surf.descriptorSize())

print(des.shape)
##########################


index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2


import numpy as np
import cv2
from matplotlib import pyplot as plt
'''
import numpy as np
from matplotlib import pyplot as plt

import cv2
def funcCheck(image1,image2):
        # Initiate SIFT detector
        print("start func Check")
        sift = cv2.xfeatures2d.SIFT_create()
        #changing from PIL to nparray to work with "detectandCompute"
        # find the keypoints and descriptors with SIFT
        img1 = np.array(image1)
        img2 = np.array(image2)
        print("after converting to np")

        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        print("after flann")

        matches = flann.knnMatch(des1,des2,k=2)
        print("after matches")
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        print("after matchesMASK")

        # ratio test as per Lowe's paper
        p=0#counter
        for i, (m, n) in enumerate(matches):
            #print("not matched")
            if m.distance < 0.7*n.distance:
                matchesMask[i] = [1, 0]
                ## Notice: How to get the index
                p=p+1
                #print("matched")
                #print(p)
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
                ## Draw pairs in purple, to make sure the result is ok
                cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 10, (255, 0, 255), -1)
                cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 10, (255, 0, 255), -1)

        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask,
                           flags = 0)

        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

        #plt.imshow(img3,),plt.show()
        return p

#img1 = cv2.imread('one.jpeg',0)          # queryImage
#img2 = cv2.imread('two.jpeg',0)

#x=funcCheck(img1,img2)
#print(x)

from PIL import Image

def FetureCount(image1):

    images = np.array(image1)
    img1 = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    return len(kp1)