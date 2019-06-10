import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from math import atan2,degrees,sqrt
#from SURF2 import DB_SCAN
from scipy.spatial import Delaunay

outputFolder=[]
def imageDeleteObject(Image,objectSize):
    test = Image.copy()
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(objectSize," object size ")
    if len(objectSize)>0:
        if(type(objectSize[0] is int)):
            print(objectSize)
            print('int')
            #test[int(objectSize[2]):int(objectSize[2]) + (int(objectSize[3]) - int(objectSize[2])),int(objectSize[0]):int(objectSize[0]) + (int(objectSize[1]) - int(objectSize[0]))] = 0
            test[objectSize[0]:objectSize[1], objectSize[2]:objectSize[3]] = 0
        else:
            for range in objectSize:
                print(objectSize)
                print('not int')
                #test[int(range[2]):int(range[2]) + (int(range[3])-int(range[2])), int(range[0]):int(range[0]) + (int(range[1])-int(range[0]))] = 0
                test[objectSize[0]:objectSize[1], objectSize[2]:objectSize[3]] = 0


    cv2.imwrite('output/imageaftercropped.jpg', test)
    return test
def getFolder(folder):
    outputFolder.append(folder)
    print(outputFolder[0])

def reset():
    outputFolder.clear()
def returnCroppedParts(imagetosend,imagetotakeclustersfrom,dict2,dict):

    cnt=0
    l_img = imagetosend
    for i in dict2.keys():
        r = dict.get(i)
        if r[2]>0 and r[3]>0:
            #coor1=(  int(r[0]-(r[2]))  ,  int(r[0]+(r[2])))
            #coor2 = (int(r[1] - (r[3] )), int(r[1] + (r[3] )))
            #if coor1[0]>1 and coor2[0]>1 and coor1[1]>1 and coor2[1]>1:
                crp=imagetotakeclustersfrom[int(r[0]):int(r[0])+int(r[2]),int(r[1]):int(r[1])+int(r[3])]
                cv2.imwrite(outputFolder[0]+'/cropt'+str(cnt)+'.jpg',crp)
                #imagetosend[dict2.get(i)]=resize(imagetotakeclustersfrom.dict.get(i))
                k=dict2.get(i)
                dim=(k[2],k[3])
                #dim=(2,2)
                #print("before")
                #print(crp.shape[0])
                #print(crp.shape[1])

                if (dim[0] > 1 and dim[1]>1 and (crp.shape[0]>0 and crp.shape[1]>0)):
                    #print(dim)
                    #crp=cv2.resize(crp,dim,interpolation=cv2.INTER_AREA)
                    #print("after")
                    #print(crp.shape[0])
                    #print(crp.shape[1])
                    #print(smallemala)
                    smallemala = dict2.get(i)
                    x_offset=int(smallemala[1])
                    y_offset=int(smallemala[0])
                    #s_img=crp
                    print("small lemala", smallemala)
                    print("s_img.shape 0",crp.shape[0])
                    print("s_img.shape[1]",crp.shape[1])
                    ss_img=cv2.resize(crp, (int(smallemala[3]), int(smallemala[2])),interpolation=cv2.INTER_AREA)
                    #print(cnt)
                    print("x_offset ",x_offset)
                    print("y_offset ",y_offset)
                    print("s_img.shape[0] ",ss_img.shape[0])
                    print("s_img.shape[1] ",ss_img.shape[1])
                    print()
                    print("l_img.shape[0] ",l_img.shape[0])
                    print("l_img.shape[1] ",l_img.shape[1])
                    #print(imagetosend.shape[0])
                    #print(l_img.shape[1])
                    '''
                    if x_offset + s_img.shape[0]<crp.shape[0]:
                        if (y_offset + s_img.shape[1] < crp.shape[1]):
                            l_img[x_offset: x_offset + s_img.shape[0] + 1, y_offset:y_offset + s_img.shape[1] + 1] = s_img
                        else:
                            l_img[x_offset : x_offset + s_img.shape[0]+1, y_offset:y_offset + s_img.shape[1]] = s_img
                    else:
                        if y_offset + s_img.shape[1] < crp.shape[1]:
                            l_img[x_offset : x_offset + s_img.shape[0], y_offset:y_offset + s_img.shape[1]+1] = s_img
                        else:
                            print("not small")
                            print(s_img.shape[0])
                            print(s_img.shape[1])
                            print(x_offset, x_offset + s_img.shape[0])
                            print(y_offset, y_offset + s_img.shape[1])
                            print("end not small")
                    '''
                    l_img[ y_offset: y_offset + int(smallemala[2]), x_offset:x_offset + int(smallemala[3])] = ss_img
                    #s_img.shape[0]
        cnt = cnt + 1
    cv2.imwrite(outputFolder[0]+'/profetOmriGOOD.jpg', l_img)
    return l_img

def returnCroppedParts2(imagetosend,imagetotakeclustersfrom,dict3,dict):
    cnt=0
    l_img = imagetosend
    for i in dict3.keys():
        r = dict.get(i)
        if r[2]>0 and r[3]>0:
            #coor1=(  int(r[0]-(r[2]))  ,  int(r[0]+(r[2])))
            #coor2 = (int(r[1] - (r[3] )), int(r[1] + (r[3] )))
            #if coor1[0]>1 and coor2[0]>1 and coor1[1]>1 and coor2[1]>1:
                crp=imagetotakeclustersfrom[int(r[0]):int(r[0])+int(r[2]),int(r[1]):int(r[1])+int(r[3])]
                cv2.imwrite(outputFolder[0]+'/bad0'+str(cnt)+'.jpg',crp)
                #imagetosend[dict2.get(i)]=resize(imagetotakeclustersfrom.dict.get(i))
                k=dict3.get(i)
                dim=(k[2],k[3])
                #dim=(2,2)
                #print("before")
                #print(crp.shape[0])
                #print(crp.shape[1])

                if (dim[0] > 1 and dim[1]>1 and (crp.shape[0]>0 or crp.shape[1]>0)):
                    #print(dim)
                    #crp=cv2.resize(crp,dim,interpolation=cv2.INTER_AREA)
                    #print("after")
                    #print(crp.shape[0])
                    #print(crp.shape[1])
                    #print(smallemala)
                    smallemala = dict3.get(i)
                    x_offset=int(smallemala[1])
                    y_offset=int(smallemala[0])
                    #s_img=crp
                    print("small lemala", smallemala)
                    print("s_img.shape 0",crp.shape[0])
                    print("s_img.shape[1]",crp.shape[1])
                    ss_img=cv2.resize(crp, (int(smallemala[3]), int(smallemala[2])),interpolation=cv2.INTER_AREA)
                    #print(cnt)
                    print("x_offset ",x_offset)
                    print("y_offset ",y_offset)
                    print("s_img.shape[0] ",ss_img.shape[0])
                    print("s_img.shape[1] ",ss_img.shape[1])
                    print()
                    print("l_img.shape[0] ",l_img.shape[0])
                    print("l_img.shape[1] ",l_img.shape[1])
                    #print(imagetosend.shape[0])
                    #print(l_img.shape[1])
                    '''
                    if x_offset + s_img.shape[0]<crp.shape[0]:
                        if (y_offset + s_img.shape[1] < crp.shape[1]):
                            l_img[x_offset: x_offset + s_img.shape[0] + 1, y_offset:y_offset + s_img.shape[1] + 1] = s_img
                        else:
                            l_img[x_offset : x_offset + s_img.shape[0]+1, y_offset:y_offset + s_img.shape[1]] = s_img
                    else:
                        if y_offset + s_img.shape[1] < crp.shape[1]:
                            l_img[x_offset : x_offset + s_img.shape[0], y_offset:y_offset + s_img.shape[1]+1] = s_img
                        else:
                            print("not small")
                            print(s_img.shape[0])
                            print(s_img.shape[1])
                            print(x_offset, x_offset + s_img.shape[0])
                            print(y_offset, y_offset + s_img.shape[1])
                            print("end not small")
                    '''
                    l_img[ y_offset: y_offset + int(smallemala[2]), x_offset:x_offset + int(smallemala[3])] = ss_img
                    #s_img.shape[0]
        cnt = cnt + 1
    cv2.imwrite(outputFolder[0]+'/profetOmriBAD.jpg', l_img)
    return l_img
def returnCroppedParts2try(imagetosend,imagetotakeclustersfrom,dict2,dict):
    arr=[]
    for i in dict2.keys():
        r=dict2.get(i)
        q=dict.get(i)
        if r[2]==0 or r[3] ==0 or q[2]==0 or q[3] ==0:
          arr.append(i)
    for i in arr:
        dict2.pop(i)
        dict.pop(i)
    for i in dict2.keys():
        r = dict2.get(i)
        q = dict.get(i)
        #print(i,r,"   ",q)
    cnt=0
    for i in dict2.keys():
        r = dict2.get(i)
        q = dict.get(i)
        #print(i,r,"   ",q)
        dim = (r[2]*2, r[3]*2)
        rtopleftx=r[0]-r[2]
        rtoplefty=r[1]-r[3]
        rh=r[2]*2
        rw=r[3]*2
        qtopleftx = q[0] - q[2]
        qtoplefty = q[1] - q[3]
        qh = q[2] * 2
        qw = q[3] * 2
        crp=imagetotakeclustersfrom[int(qtoplefty):int(qtoplefty+qh), int(qtopleftx):int(qtopleftx+qw)]
        cv2.imwrite(outputFolder[0]+'/croptGGG' + str(cnt) + '.jpg', crp)
        if dim[0]==0 or dim[1]==0 or crp.shape[0]==0 or crp.shape[1]==0 or crp.shape[2]==0:
            print("something is zero",cnt)
        else:
            print(crp.shape)
            crp=cv2.resize(crp,dim,interpolation=cv2.INTER_AREA)
            print(dim,crp.shape)
            imagetosend[ int(rtopleftx):int(rtopleftx)+rw,int(rtoplefty):int(rtoplefty)+rh]=crp
            cv2.imwrite(outputFolder[0]+'/cropt' + str(cnt) + '.jpg', crp)
        cnt=cnt+1
def returnCroppedParts3try(imagetosend,imagetotakeclustersfrom,dict2,dict):
    arr=[]
    for i in dict2.keys():
        r=dict2.get(i)
        q=dict.get(i)
        if r[2]==0 or r[3] ==0 or q[2]==0 or q[3] ==0:
          arr.append(i)
    for i in arr:
        dict2.pop(i)
        dict.pop(i)
    cnt=0
    for i in dict2.keys():
        r = dict2.get(i)
        q = dict.get(i)
        #print(i,r,"   ",q)
        dim = (int(r[2]), int(r[3]))
        rtopleftx=r[0]
        rtoplefty=r[1]
        rh=r[2]
        rw=r[3]
        qtopleftx = q[0]
        qtoplefty = q[1]
        qh = q[2]
        qw = q[3]
        crp=imagetotakeclustersfrom[int(qtoplefty):int(qtoplefty+qh), int(qtopleftx):int(qtopleftx+qw)]
        cv2.imwrite(outputFolder[0]+'/croptSSS' + str(cnt) + '.jpg', crp)
        if dim[0]==0 or dim[1]==0 or crp.shape[0]==0 or crp.shape[1]==0 or crp.shape[2]==0:
            print("something is zero",cnt)
        else:
            print(crp.shape)
            crp=cv2.resize(crp,dim,interpolation=cv2.INTER_AREA)
            print(dim,crp.shape)
            print("rh ", rh,"rw ",rw, "shape ",crp.shape[0],crp.shape[1])
            print("after cast ",int(rw),int(rh))
            imagetosend[ int(rtopleftx):int(rtopleftx)+int(rw),int(rtoplefty):int(rtoplefty)+int(rh)]=crp
            cv2.imwrite('output/cropt' + str(cnt) + '.jpg', crp)
        cnt=cnt+1
        cv2.imwrite(outputFolder[0]+'/profetOmri.jpg', imagetosend)
def AngleBtw2Points(pointA, pointB):
  changeInX = pointB[0] - pointA[0]
  changeInY = pointB[1] - pointA[1]
  return degrees(atan2(changeInY,changeInX))
def distanceAndAngle(arrayOfClusters):
    centers = []
    for cluster in arrayOfClusters:
        minY, maxY, minX, maxX = corMinMax(cluster)
        newX,newY,a,b = SizeandCenter(minY, maxY, minX, maxX)
        center=newX,newY
        centers.append(center)
    arrayOfDistanceAndAngle=[]
    n = len(centers)
    for i in range(0,n):
        smallArray=[]
        for j in range(0, n):
            angle=AngleBtw2Points(centers[i],centers[j])
            distance = sqrt(((centers[i][0] - centers[j][0]) ** 2) + ((centers[i][1] - centers[j][1]) ** 2))
            #distance = math.sqrt(((p1[0] - p2[0]) * 2) + ((p1[1] - p2[1]) * 2))
            distanceAndangle=j,angle,distance
            smallArray.append(distanceAndangle)
        tupletoAdd=(i,smallArray,centers[i])
        arrayOfDistanceAndAngle.append(tupletoAdd)
    return arrayOfDistanceAndAngle
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
    db = DBSCAN(eps=epsilon, min_samples=8).fit(AKP)
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
        #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #         markeredgecolor='k', markersize=14)
        xy = AKP[class_member_mask & ~core_samples_mask]
        counter = counter + (len(xy))
        #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #         markeredgecolor='k', markersize=6)
        if (k != -1):
            biglist.append(new_list)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    #plt.show()
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
    return (p1 / len(p2)),image2coordinates,arrayKP
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
def topleftAndSizes(minY,maxY,minX,maxX):
    TopLeft=(minY,minX)
    sizes=(maxY-minY,maxX-minX)
    print(TopLeft,sizes)
    return(TopLeft[0],TopLeft[1],sizes[0],sizes[1])
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
    cv2.imwrite(outputFolder[0]+'/test.jpg', test)
    for range in partsList:
        test[ int(range[0] - range[3]) : int(range[0] + range[3]),int(range[1] - range[2]) : int(range[1] + range[2]) ] = 0
    cv2.imwrite(outputFolder[0]+'/test2.jpg', test)
    return test

def imageDeleteParts2(Image, partsList):
    checker=0

    for range in partsList:
        check=Image[int(range[0]):int(range[0])+int(range[2]),int(range[1]):int(range[1])+int(range[3])]
        cv2.imwrite(outputFolder[0]+'/check'+str(checker)+'.jpg', check)
        Image[int(range[0]):int(range[0])+int(range[2]),int(range[1]):int(range[1])+int(range[3])] = 0
        checker=checker+1
    cv2.imwrite(outputFolder[0]+'/imageaftercropped.jpg', Image)
    return Image


def imageDeleteParts3(Image, partsList):
    test = Image.copy()
    #checker = 0
    for range in partsList:
        #check = Image[int(range[0]):int(range[0]) + int(range[2]), int(range[1]):int(range[1]) + int(range[3])]
        #cv2.imwrite('output/check' + str(checker) + '.jpg', check)
        test[int(range[0]):int(range[0]) + int(range[2]), int(range[1]):int(range[1]) + int(range[3])] = 0
        #checker = checker + 1
    cv2.imwrite(outputFolder[0]+'/imageaftercropped.jpg', test)
    return test


def imageDeleteParts2seconduse(Image, partsList):
    checker=0
    for range in partsList:
        check=Image[int(range[0]):int(range[0])+int(range[2]),int(range[1]):int(range[1])+int(range[3])]
        cv2.imwrite(outputFolder[0]+'/checkGGG'+str(checker)+'.jpg', check)
        Image[int(range[0]):int(range[0])+int(range[2]),int(range[1]):int(range[1])+int(range[3])] = 0
        checker=checker+1
    cv2.imwrite(outputFolder[0]+'/imageaftercropped.jpg', Image)
    return Image

def makeDictforOriginalClusters(clusters): # make dictionary for server image
    dict={}#dict of dictionary and flag, for first photo
    for i in range(0,len(clusters)):
        minY, maxY, minX, maxX = corMinMax(clusters[i])
        #SizeCenter = SizeandCenter(minY, maxY, minX, maxX)
        SizeCenter = topleftAndSizes(minY, maxY, minX, maxX)
        dict.update({i:(SizeCenter)})
    return dict

def makeDictforGoodClusters(arrayOfGoodclusters,flagsOfGoodClusters): # make dictionary for Good clusters
    dict2 = {}
    print("flags")
    # create dict2 for client image
    for i in range(0, len(arrayOfGoodclusters)):
        minY, maxY, minX, maxX = corMinMax(arrayOfGoodclusters[i])
        # SizeCenter = SizeandCenter(minY, maxY, minX, maxX)
        SizeCenter = topleftAndSizes(minY, maxY, minX, maxX)
        z = flagsOfGoodClusters[i]
        dict2.update({z: SizeCenter})
        print(i, z)
    return dict2

def makeDictforBadClusters(arrayOfBadclusters,flagsOfBadClusters):
    dict3={}
    print("Badflags")
    #create dict2 for client image
    print("bad clusters are: ",arrayOfBadclusters)
    for i in range (0,len(arrayOfBadclusters)):
        minY, maxY, minX, maxX = corMinMax(arrayOfBadclusters[i])
        #SizeCenter = SizeandCenter(minY, maxY, minX, maxX)
        SizeCenter=topleftAndSizes(minY, maxY, minX, maxX)
        h=flagsOfBadClusters[i]
        dict3.update({h:SizeCenter})
        print(i,h)
    return dict3

def readImagesToMakeCommonImage(arrayServerImgs):

    arrayimg=[]
    for img in arrayServerImgs:
        arrayimg.append(cv2.imread(img))

    return arrayimg
def clustersOfCroppedImage(image1,dbscan_epsilon):
    #sift = cv2.xfeatures2d.SIFT_create()

    cv2.imwrite(outputFolder[0]+'/imagecheck.jpg', image1)
    surf = cv2.xfeatures2d.SURF_create()
    img1 = np.array(image1)
    kp, des = surf.detectAndCompute(img1, None)
    dictionary = CreateDict(kp, des)
    clusters = DB_SCAN(kp,dbscan_epsilon)

    return clusters,dictionary,kp,des
def funcCheck1(image1, image2):
    # Initiate SIFT detector
    #print("funcheck1")
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
    #print("funccheck1 ,matches,number of kp first image",p, " ",len(kp1))
    return p,okp,odes  # returns number of best matches,and all keypoints of first img
def funcCheck2(kp,des, image2):
    #print("funccheck 2")
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
    #print("funccheck2 results ",p,len(kp1))
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
        #print("kp1",len(kp1))
        #print("kp2",len(kp2))
        #print("matches", p)
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
def makegoodclusters(clusters,dictionary,image,threshold,NClustersWObjects,listOfNumbers):
    arrayOfGoodclusters=[]
    arrayOfBadclusters=[]
    newListOfNumbers=[]
    counter=0
    count_originals=0
    flags=[]
    flags2 = []
    for cluster in clusters:
        #print("cluster is: ", cluster)
        print("cluster Number: ",counter)
        y,img2coords,getarraykp=checkCluster(cluster,dictionary,image)
        print("grade of cluster: ",y)
        #addp2 instead of cluster
        print("size of cluster ",len(getarraykp))
        print("img2coords ",len(img2coords))
        print(" % is ",y)
        if y>threshold:
          #print("length of clusters ",y,len(img2coords))
          arrayOfGoodclusters.append(img2coords)
          flags.append(counter)
          if(counter>=NClustersWObjects):
              newListOfNumbers.append(listOfNumbers[counter-NClustersWObjects])
          else:
                count_originals = count_originals + 1
        else: #option- add a condition , of the number of kp
            if len(getarraykp)>2 and len(img2coords)>2:
                print(img2coords, "HII")
                arrayOfBadclusters.append(img2coords)
                flags2.append(counter)
            else:
                print(img2coords, "wrong")
        counter=counter+1
    print("Number of good clusters: ", len(arrayOfGoodclusters))
    print("Number of bad clusters: ", len(arrayOfBadclusters))
    return arrayOfGoodclusters,flags,arrayOfBadclusters,flags2,newListOfNumbers,count_originals
    #return arrayOfGoodclusters

def makecroppedimage(arrayOfGoodclusters,image,newListOfNumbers,count_originals,range_list):
    count=0
    sizes=[]
    range_list_new=[]
    for cluster in arrayOfGoodclusters:
        if(count<count_originals):
            minY, maxY, minX, maxX = corMinMax(cluster)
            #SizeCenter = SizeandCenter(minY, maxY, minX, maxX)
            SizeCenter=topleftAndSizes(minY, maxY, minX, maxX)
            #sizes.append(SizeCenter)
            sizes.append(SizeCenter)
        else:
            range_list_new.append(range_list[newListOfNumbers[count-count_originals]])
    print("$$$$$$$$$$$")
    print("len of size is 34:", len(sizes))
    #croppedimage= imageDeleteParts2(image,sizes)
    croppedimage = imageDeleteParts3(image,sizes)######maybe we should swap with the line below
    croppedimage_new = imageDeleteObject(croppedimage, range_list_new)
    cv2.imwrite(outputFolder[0]+'/testguy.jpg',croppedimage_new)
    return croppedimage_new
def makecroppedimageseconduse(arrayOfGoodclusters,image):
    sizes=[]
    for cluster in arrayOfGoodclusters:
        minY, maxY, minX, maxX = corMinMax(cluster)
        #SizeCenter = SizeandCenter(minY, maxY, minX, maxX)
        SizeCenter=topleftAndSizes(minY, maxY, minX, maxX)
        #sizes.append(SizeCenter)
        sizes.append(SizeCenter)
    print("$$$$$$$$$$$")
    print("len of size is ?:", len(sizes))
    croppedimage= imageDeleteParts2seconduse(image,sizes)
    cv2.imwrite(outputFolder[0]+'/testguy.jpg',croppedimage)
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
'''
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
'''

import cv2
import numpy as np
from scipy.optimize import fsolve
def findcenters(clusters):
    centers=[]
    print("clusteRSSSSS: ")
    for i in clusters:
        if len(i)>0:
            minY, maxY, minX, maxX=corMinMax(i)
            center=((minX+maxX)/2,(minY+maxY)/2)
            centers.append(center)
        else:
            centers.append(center)#patch
    return centers
def findextremeclusteres(badclusters,indexes):
    centers=findcenters(badclusters)
    minX=centers[0][0]
    maxX=0
    minY=centers[0][1]
    indexminX=-1
    indexmaxX=-1
    indexminY=-1
    for j,index in zip(centers,indexes):
        #print("centers")
        #print(j[0])
        if j[0]>maxX:
            indexmaxX=index
            maxX=j[0]
        else:
            if j[0]<minX:
                indexminX = index
                minX = j[0]
        if j[1]<minY:
            indexminY = index
            minY = j[1]

    return indexminX,indexmaxX,indexminY
def fourth(indexOne,indexTwo,indexThree):
    a= indexOne
    b = indexTwo
    c = indexThree
    return a,b,c

def firstFuncCheck(FirstImage):
    surf = cv2.xfeatures2d.SURF_create()
    img1 = np.array(FirstImage)
    kp, des = surf.detectAndCompute(img1, None)
    return kp,des

def IntersectOfImages2(listOfObjects,newSortedArrayimg):
    i=0
    j=0
    listOfMatches = []
    for i in range (0,len(listOfObjects)):
        arraykp = listOfObjects[i][0]
        arraydes = listOfObjects[i][1]
        num = len(arraykp)
        for j in range (1,len(newSortedArrayimg)):
            x,arraykp,arraydes = funcCheck2(arraykp,arraydes,newSortedArrayimg[j])
        ratio = x/num
        matchesTupel = (arraykp,arraydes,ratio)
        listOfMatches.append(matchesTupel)
    return listOfMatches

def keyOfObject(range_list, kp, des):
    listOfObjects = []
    for i in range(0, len(range_list)):
        descriptorOfObjects = []
        keyOfObjects = []
        for j, k in zip(kp, des):
            # for cor in array_Kp_pt:
            if ((j.pt[0] >= range_list[i][2]) and (j.pt[0] <= range_list[i][3])):
                if ((j.pt[1] >= range_list[i][0]) and (j.pt[1] <= range_list[i][1])):
                    keyOfObjects.append(j)
                    descriptorOfObjects.append(k)
                    object_tupel = (keyOfObjects, descriptorOfObjects)
        listOfObjects.append(object_tupel)
    return listOfObjects

def divideArrayOfIMG(newSortedArrayimg, threshold2):
    flag = [1] * len(newSortedArrayimg)
    allarray = []
    for i in range(0, len(newSortedArrayimg)):
        if flag[i] == 1:
            array = []
            array.append(newSortedArrayimg[i])
            for j in range(1, len(newSortedArrayimg)):
                if flag[j] == 1:
                    temp, doesnotmatter1, doesnotmatter2 = funcCheck1(newSortedArrayimg[i], newSortedArrayimg[j])
                    if temp > threshold2:
                        array.append(newSortedArrayimg[j])
                        flag[j] = 0
                    else:
                        print("nothing")

                else:
                    print("nothing")
            allarray.append(array)
    return allarray

def matchedObjects(listOfMatches, range_list, croped):
    new_listOfMatches = []
    listOfNumbers = []
    for i in range(0, len(listOfMatches)):
        if (listOfMatches[i][2] > 0.4):
            croped = imageDeleteObject(croped, range_list[i])
            # cv2.imwrite(outputFolder + '/croppedBoris'+str(i)+'.jpg', croped)
            listOfNumbers.append(i)
            t_list = (listOfMatches[i][0], listOfMatches[i][1])
            new_listOfMatches.append(t_list)
    return croped, new_listOfMatches, listOfNumbers

def updateDict(dictionary,new_listOfMatches):
    for i in new_listOfMatches:
        for j, k in zip(i[0], i[1]):
            dictionary.update({j.pt: (j, k)})
    return dictionary

def updateCluster(kp_1,dbscan_epsilon,new_listOfMatches):
    clusters = DB_SCAN(kp_1, dbscan_epsilon) #clustering the kp according to coords
    NClustersWObjects = len(clusters)
    objectS_list = []
    for i in new_listOfMatches:
        for j in i[0]:
            objectS_list.append(j.pt)
    clusters.append(objectS_list)
    return clusters,NClustersWObjects

def updateCluster2(kp_1,Newclusters2,new_listOfMatches):
    #clustering the kp according to coords
    NClustersWObjects = len(Newclusters2)
    objectS_list = []
    for i in new_listOfMatches:
        for j in i[0]:
            objectS_list.append(j.pt)
    Newclusters2.append(objectS_list)
    return Newclusters2,NClustersWObjects