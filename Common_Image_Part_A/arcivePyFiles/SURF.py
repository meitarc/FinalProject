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
import pandas as pd
import cv2
def funcCheck(image1,image2):
        # Initiate SIFT detector
        #print("start func Check")
        sift = cv2.xfeatures2d.SIFT_create()
        #changing from PIL to nparray to work with "detectandCompute"
        # find the keypoints and descriptors with SIFT
        img1 = np.array(image1)
        img2 = np.array(image2)
        #print("after converting to np")

        kp1, des1 = sift.detectAndCompute(img1,None)

        #############################
        #print(kp1[0])
        point=kp1[0]
        #print(point.class_id)
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        print(temp)
        temp_feature = cv2.KeyPoint(temp[0][0], temp[0][1], temp[1], temp[2], temp[3], temp[4], temp[5])
        print(temp_feature)
        #print(temp_feature)
        #print(type(des1))
        #print(type(des1[0]))



        x,y=kp1[0].pt
        #print(type(x))
        #print(y)
        #print(type(kp1[0]))
        #print(type(des1))
        #print(kp1[0])
        #print(kp1[0])
        dfkeypoint=pd.DataFrame(kp1)
        #print(type(dfkeypoint))
        dfkeypoint.to_csv('keyCsv.csv', index=False)
        fromCsvKey=pd.read_csv("C:/Users/Meitar/קבצי לימודים/שנה ג/פרויקט גמר/FinalProject/Common_Image_Part_A/keyCsv.csv")
        #print(type(fromCsvKey))

        listKey=fromCsvKey.values.tolist()
        #print(type(listKey))
        #print("---------------------")
        #print(listKey)
        #print(fromCsvKey)
        #print(type(des1))
        #print(des1)
        #print("----------------------------------------------------")
        df=pd.DataFrame(des1)
        #print(df)
        df.to_csv('desCsv.csv', index=False)
        #print(des1[0])
        fromCsv=pd.read_csv("C:/Users/Meitar/קבצי לימודים/שנה ג/פרויקט גמר/FinalProject/Common_Image_Part_A/desCsv.csv")
        #print(fromCsv)
        numpy_matrix = fromCsv.values
        #print(numpy_matrix)
        #print(type(numpy_matrix))



        #######################################
        kp2, des2 = sift.detectAndCompute(img2,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        #print("after flann")

        matches = flann.knnMatch(des1,des2,k=2)
        #print("after matches")
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        #print("after matchesMASK")

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

img1 = cv2.imread('try0.jpg',0)          # queryImage
img2 = cv2.imread('try1.jpg',0)

x=funcCheck(img1,img2)
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



