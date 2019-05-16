import cv2
import numpy as np
from sklearn.cluster import DBSCAN

#from SURF2 import DB_SCAN
from scipy.spatial import Delaunay

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges
def find_edges_with(i, edge_set):
    i_first = [j for (x,j) in edge_set if x==i]
    i_second = [j for (j,x) in edge_set if x==i]
    return i_first,i_second


def croppedmatchingareas(clientimagepath, goodclustersarray):
    #image = cv2.imread(clientimagepath)
    image = clientimagepath

    # create a mask with white pixels
    mask = np.ones(image.shape, dtype=np.uint8)
    mask.fill(255)
    Newclusters = []
    for i in goodclustersarray:
        for j in i:
            Newclusters.append(j)

    arrays = np.array(Newclusters)
    # points to be cropped
    roi_corners = np.array([arrays], dtype=np.int32)
    # fill the ROI into the mask
    cv2.fillPoly(mask, roi_corners, 0)

    # The mask image
    cv2.imwrite('image_masked.jpg', mask)

    # applying th mask to original image
    masked_image = cv2.bitwise_or(image, mask)

    # The resultant image
    cv2.imwrite('newcropped.jpg', masked_image)
    return masked_image

def extractKeyPt(kp1):
    array_Kp_pt = []
    for kp in kp1:
        array_Kp_pt.append(kp.pt)
    return array_Kp_pt

def DB_SCAN(keypointsArray,epsilon):
    Arraykeypoints = extractKeyPt(keypointsArray)
    # Generate sample data
    AKP = np.array(Arraykeypoints)
    # Compute DBSCAN
    db = DBSCAN(eps=epsilon, min_samples=10).fit(AKP)
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
            #new_list.append(k)
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


def IntersectOfImages(arrayOfimages):
    x, arraykp, arraydes = funcCheck1(arrayOfimages[0], arrayOfimages[1])
    for i in range (2,len(arrayOfimages)):
        x,arraykp,arraydes = funcCheck2(arraykp,arraydes,arrayOfimages[i])
    print("IntersectOfImages")
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
    p1, p2,image2coordinates = clientFuncCheck(arrayKP, arrayDES, image,1)
    return (p1 / len(p2)),image2coordinates

def corMinMax(cluster):
    maxX = 0
    maxY = 0
    #print(cluster)
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
    cv2.imwrite('output/test.jpg', test)
    for range in partsList:
        test[ int(range[0] - range[3]) : int(range[0] + range[3]),int(range[1] - range[2]) : int(range[1] + range[2]) ] = 0
    cv2.imwrite('output/test2.jpg', test)
    return test

def readImagesToMakeCommonImage():
    img3 = cv2.imread("source/101.jpg")
    img4 = cv2.imread("source/102.jpg")
    img5 = cv2.imread("source/103.jpg")
    img6 = cv2.imread("source/104.jpg")
    arrayimg=[img3,img4,img5,img6]
    return arrayimg

def clustersOfCroppedImage(image1):
    #sift = cv2.xfeatures2d.SIFT_create()

    cv2.imwrite('output/imagecheck.jpg', image1)
    surf = cv2.xfeatures2d.SURF_create()
    img1 = np.array(image1)
    kp, des = surf.detectAndCompute(img1, None)
    dictionary = CreateDict(kp, des)
    clusters = DB_SCAN(kp,20)

    return clusters,dictionary

def funcCheck1(image1, image2):
    # Initiate SIFT detector
    print("funcheck1")
    surf = cv2.xfeatures2d.SURF_create()
    #sift = cv2.xfeatures2d.SIFT_create()
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
        if m.distance < 0.95 * n.distance:
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
    print("funccheck1 ,matches,number of kp first image",p, " ",len(kp1))
    return p,okp,odes  # returns number of best matches,and all keypoints of first img

def funcCheck2(kp,des, image2):
    print("funccheck 2")
    # Initiate SIFT detector
    #surf = cv2.SURF(400)
    #sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
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
        if m.distance < 0.95 * n.distance:
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
    print("funccheck2 results ",p,len(kp1))
    return p,okp,odes

def clientFuncCheck(one, two, image2,flag):
    #print("my funcCheck 2, for each cluster:")
    if flag==1:
        # Initiate SIFT detector
        surf = cv2.xfeatures2d.SURF_create()
        #surf = cv2.xfeatures2d.SURF_create()
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
        img2coordnts=[]

        for i, (m, n) in enumerate(matches):

            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
                ## Notice: How to get the index
                p = p + 1
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
                img2coordnts.append(kp2[m.trainIdx].pt)
                ## Draw pairs in purple, to make sure the result is ok
                #cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 10, (255, 0, 255), -1)
                #cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 10, (255, 0, 255), -1)

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=0)

        # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

        # plt.imshow(img3,),plt.show()
        print("kp1",len(kp1))
        print("kp2",len(kp2))
        print("matches", p)
        return p, kp1,img2coordnts  # returns number of best matches,and all keypoints of first img
    else:
        # Initiate SIFT detector
        surf = cv2.xfeatures2d.SURF_create()
        # surf = cv2.xfeatures2d.SURF_create()
        # changing from PIL to nparray to work with "detectandCompute"
        # find the keypoints and descriptors with SIFT
        img2 = np.array(image2)

        kp1, des1 = one, two

        kp2, des2 = surf.detectAndCompute(img2, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=1)
        search_params = dict(checks=10)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # http://answers.opencv.org/question/35327/opencv-and-python-problems-with-knnmatch-arguments/
        matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        p = 0  # counter
        for i, (m, n) in enumerate(matches):

            if m.distance < 0.95 * n.distance:
                matchesMask[i] = [1, 0]
                ## Notice: How to get the index
                p = p + 1
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
                ## Draw pairs in purple, to make sure the result is ok
                # cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 10, (255, 0, 255), -1)
                # cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 10, (255, 0, 255), -1)

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=0)

        # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

        # plt.imshow(img3,),plt.show()
        return p, kp1  # returns number of best matches,and all keypoints of first img


def makegoodclusters(clusters,dictionary,image,threshold):
    arrayOfGoodclusters=[]
    counter=0

    for cluster in clusters:
        #print("cluster is: ", cluster)
        print("cluster Number: ",counter)
        y,img2coords=checkCluster(cluster,dictionary,image)
        #print("grade of cluster: ",y)
        #addp2 instead of cluster
        print(" % is ",y)
        if y>threshold:
          print("length of clusters ",y,len(img2coords))
          arrayOfGoodclusters.append(img2coords)
        counter=counter+1
    return arrayOfGoodclusters


def makecroppedimage(arrayOfGoodclusters,image):
    sizes=[]
    for cluster in arrayOfGoodclusters:
        minY, maxY, minX, maxX = corMinMax(cluster)
        SizeCenter = SizeandCenter(minY, maxY, minX, maxX)
        sizes.append(SizeCenter)
    croppedimage= imageDeleteParts(image,sizes)
    cv2.imwrite('output/testguy.jpg',croppedimage)
    return croppedimage


def sortImageByFeachers(arrayimg):
    newArrayimg = arrayimg
    list=[]
    for image in newArrayimg:
        x=FetureCount(image)
        list.append(x)
    n = len(list)
    for i in range(n):
        for j in range(0, n - i - 1):
            if list[j] < list[j + 1]:
                list[j], list[j + 1] = list[j + 1], list[j]
                newArrayimg[j],newArrayimg[j+1]=newArrayimg[j+1],newArrayimg[j]
    print(list)
    return newArrayimg


def FetureCount(image1):
    images = np.array(image1)
    img1 = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    # Initiate SIFT detector
    surf = cv2.xfeatures2d.SURF_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(img1, None)
    return len(kp1)


def tryit(image):
    images = np.array(image)
    img1 = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    from matplotlib import pyplot as plt
    img = np.array(img1)
    for i in kp1:
        # print(type(i))
        cv2.circle(img, (int(i.pt[0]), int(i.pt[1])), 10, (255, 0, 255), -1)
    plt.imshow(img, ), plt.show()
