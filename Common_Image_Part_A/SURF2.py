from SURF import *
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
# from sklearn.datasets.samples_generator import make_blobs
# from sklearn.preprocessing import StandardScaler
import cv2
import pandas as pd


def checkKeyptInDict(list, dict):
    count = 0
    count1 = 0
    for i in list:
        for j in i:
            if j in dict.keys():
                count = count + 1
            else:
                count1 = count1 + 1
    print("true is ", count)
    print(count1)


# def extractKeyPt(kp1):
#    array_Kp_pt = []
#    for kp in kp1:
#        array_Kp_pt.append(kp.pt)
#    return array_Kp_pt

def extractKeyPt1(kp1, des):
    dict_Kp_pt = {}
    for kp, ds in zip(kp1, des):
        dict_Kp_pt[kp.pt] = kp.size, kp.angle, kp.response, kp.octave, kp.class_id, ds
    return dict_Kp_pt


def DB_SCAN(keypointsArray):
    Arraykeypoints = extractKeyPt(keypointsArray)
    # Generate sample data
    AKP = np.array(Arraykeypoints)
    # Compute DBSCAN
    db = DBSCAN(eps=100, min_samples=10).fit(AKP)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    '''
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))
    '''
    # Plot result
    import matplotlib.pyplot as plt
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    biglist = []
    counter = 0
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = AKP[class_member_mask & core_samples_mask]
        counter = counter + (len(xy))
        new_list = []
        if (k != -1):
            new_list.append(k)
            for cor in xy:
                x = cor[0]
                y = cor[1]
                t = (x, y)
                new_list.append(t)
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)
        xy = AKP[class_member_mask & ~core_samples_mask]
        counter = counter + (len(xy))
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
        if (k != -1):
            biglist.append(new_list)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    return biglist


def funcCheck(image1, image2):
    print("my func")
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # changing from PIL to nparray to work with "detectandCompute"
    # find the keypoints and descriptors with SIFT
    img1 = np.array(image1)
    img2 = np.array(image2)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
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
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            ## Notice: How to get the index
            p = p + 1
            pt1 = kp1[m.queryIdx].pt
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
    return p, kp1  # returns number of best matches,and all keypoints of first img


def funcCheck2(one,two, image2):
    print("my func 222")
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # changing from PIL to nparray to work with "detectandCompute"
    # find the keypoints and descriptors with SIFT
    img2 = np.array(image2)
    kp1, des1 = one,two

    kp2, des2 = sift.detectAndCompute(img2, None)

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


def top_of_kp__from_clusters(image1):
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = np.array(image1)
    kp, des = sift.detectAndCompute(img1, None)
    print("kp len ",len(kp))
    dict_Kp_pt1 = extractKeyPt1(kp, des)
    cord_array = DB_SCAN(kp)
    clusters = pd.DataFrame(data=cord_array)
    clusters = clusters.transpose()
    # clusters.to_csv('clusters.csv',index=False)
    dfcounter = 0
    list = []
    counter_else = 0
    for col in clusters:
        listnames = []
        string = "cluster" + str(dfcounter)
        listnames.append(string)
        string2 = string + "response"
        string3= string + "descriptor"
        listnames.append(string2)
        listnames.append(string3)
        arr = []
        arrImp = []
        arrDesc=[]

        for k in clusters[col]:
            # print(k)
            if k in dict_Kp_pt1.keys():
                val = dict_Kp_pt1.get(k)
                importance = val[2]
                descriptor= val[5]
                tuple1 = (k[0], k[1], val[0], val[1], val[2], val[3], val[4])
                arr.append(tuple1)
                arrImp.append(importance)
                arrDesc.append(descriptor)
            else:
                counter_else = counter_else+1
            # tuple1 = -1
            # importance=-1
            # arr.append(tuple1)
            # arrImp.append(importance)

        dataframe = df.DataFrame(data=[arr, arrImp,arrDesc])
        dataframe = dataframe.transpose()
        dataframe.columns = listnames
        list.append(dataframe)
        dfcounter = dfcounter + 1
    print("else is ", counter_else)
    for i in range(0, len(list)):
        string = "cluster" + str(i) + "response"
        list[i].sort_values(string, inplace=True, ascending=False)
        # print(list[i])
    list2 = []
    for i in list:
        # y=int(len(i)/5)
        #print("one ",len(i))

        i = i.head(int(len(i) * (20 / 100)))
        #print("two ",len(i))
        list2.append(i)


    return list2

def tuple_to_kp(t):
    #tapleKP = (t[0], t[1], t[2], t[3], t[4], t[5], t[6])
    kp = cv2.KeyPoint(t[0], t[1], t[2], t[3], t[4], int(t[5]), int(t[6]))
    return kp

def prepareData(somelist):
    #desc_array = np.array([])
    desc_array = []
    kp_list=[]
    counter=0
    for i in somelist:
        string1='cluster'
        string2='descriptor'
        string3=string1+str(counter)+string2
        string5=string1+str(counter)
        counter=counter+1
        #desc_array = np.append(desc_array, i[string3])
        for h in i[string3]:
            desc_array.append(h)

        for j in i[string5]:
            kp_list.append(tuple_to_kp(j))
    return kp_list,desc_array
# main
# regular func check getting 2 images
image1 = cv2.imread('try0.jpg')
image2 = cv2.imread('try1.jpg')
x,kp=funcCheck(image1, image2)
print(len(kp))
print("x is :",x)
list = top_of_kp__from_clusters(image1)
kpOne,desOne=prepareData(list)
print(len(kpOne))
x2,kp2=funcCheck2(kpOne,desOne,image2)
print("x2 is :",x2)