#from SURF import funcCheck
#from SURF import FetureCount
from PIL import Image
import image_slicer
from image_slicer import join
import numpy as np
import cv2
import pandas as pd

def find_max_in_column(matrix, col, max, threshold):
    img = Image.open("Black_Image.jpg")
    name = "no_max"
    for e in matrix[col]:
        print("tuple1 of col ", col, " is e: ", e, "and #0 is: ", e[0])
        if e[0] > threshold:
            if e[0] > max:
                print("got new max is :", e[0])
                max = e[0]
                img = e[1]

    return max, img


def FetureCount(image1):
    images = np.array(image1)
    img1 = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    return len(kp1)

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
    #kp1 = Load_Keypoint('keysDF.csv')
    #des1 = load_Descripto('desCsv.csv')

    kp2, des2 = sift.detectAndCompute(img2, None)

    #print(des2[0])
    #print(len(des2))

    #print(des1[0])
    #print(len(des1))

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # print("after flann")
    #http://answers.opencv.org/question/35327/opencv-and-python-problems-with-knnmatch-arguments/
    matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)

    #matches = flann.knnMatch(des1, des2, k=2)
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




def buildingPicArray(string,num):
    AtoC = []
    for i in range(0,num):
        mystring = string
        mystring=mystring+str(i)+".jpg"
        ImTestA = Image.open(mystring)
        AtoC.append(ImTestA)
        hashmap.update({mystring: ImTestA})
    arraytoReturn = []  # load array of pics
    for image in AtoC:
        arraytoReturn.append(image)
    return arraytoReturn

def Split(IMG, x):  # that split and image to x pieces, return an array of images
    return image_slicer.slice(IMG, x)

def MakeIMG(arraypic):  # that get an array of images and size and return a complete image
    image = join(arraypic)
    image.save('joined image.png')
    return image






def CALC(img1, img2):  # that calc the % of similarity(Get 2 pics)
    y = FetureCount(img1)
    open_cv_image = np.array(img1)
    open_cv_image2 = np.array(img2)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    open_cv_image2 = cv2.cvtColor(open_cv_image2, cv2.COLOR_BGR2GRAY)

    x = funcCheck(open_cv_image, open_cv_image2)  # move func check to flann file)
    if x != 0:
        return (x / y)
    else:
        return 0

def makeNewMergedIMG(arrayofImg,img2,threshold):
    ArrayIMG2 = []
    countOfGood =0
    for k in arrayofImg:
            tempo=CALC(k.image, img2)
            if tempo > threshold:
                ArrayIMG2.append(k)
                countOfGood=countOfGood+1
            #else:
                #print("tempo is: ",tempo)
    if len(ArrayIMG2)>0:
        imageMerged = MakeIMG(ArrayIMG2)
        return imageMerged,countOfGood
    else:
        return img2,0



def scoreOfSplits(array,sizeofarray,parts):
    length = sizeofarray
    matrix=[]
    for i in range(0, length):
        row =[]
        print("another row ", i)
        for j in range(0, (length)):
            print("another col ", j)
            if i != j or array[i]!=array[j]:
                array[i].save("newSCORE.jpg")
                splitImg = Split("newSCORE.jpg", parts)
                merged, counter_of_good = makeNewMergedIMG(splitImg, array[j], threshold=0.05)
                array[j].save("newSCORE2.jpg")
                splitImg2 = Split("newSCORE2.jpg", parts)
                merged2, counter_of_good2 = makeNewMergedIMG(splitImg2, array[i], threshold=0.05)
                if(counter_of_good>counter_of_good2):
                    tupletoadd =(counter_of_good,merged)
                else:
                    tupletoadd =(counter_of_good2,merged2)
                row.append(tupletoadd)
            else:
                tupletoadd=(0,0)
                row.append(tupletoadd)
        matrix.append(row)
    return matrix


def findMaxInMatrixForSplit(matrix,threshold): #return tuple of index in matrix of highest score, the index of the 2 images in the array
    max=0
    tuple1=(0,0)
    for col in list(matrix):
        print(threshold)
        #print(matrix[col].max()[0])
        print("after")
        max1,img1=find_max_in_column(matrix,col,max,threshold)
        if (max1) > threshold:
            if (max1) > max:
                max = max1
                img = img1
                tuple1 = (max, img)
                str = col
    print("after loop")
    if tuple1[0]==0:
        return col, col, tuple1[0], tuple1[1]
    else:
        row = matrix.index[matrix[str] == tuple1][0]
        return row,str,tuple1[0],tuple1[1]

def addColumToMatrixToSplit(matrix,addedMerged,arrayofpictures,parts,counter): # not used at the moment, adding another row/colum to a matrix
    print("adding")
    #length = len(matrix)
    row=[]
    string="merged"
    stringy=string+str(counter)+".jpg"
    addedMerged.save(stringy)
    dict = {string: addedMerged}
    hashmap.update(dict)
    split1 = Split(stringy, parts)
    index = -1
    for i in arrayofpictures[:]:
        index=index+1
        i.save("i.jpg")
        split2 = Split("i.jpg", parts)
        img1,score1=makeNewMergedIMG(split1, i,threshold=0.05)
        img2,score2=makeNewMergedIMG(split2,addedMerged,threshold=0.05)
        if score1>score2:
            tuple1 = (score1,img1)
        else:
            tuple1 = (score2,img2)
        row.append(tuple1)
    tup=(0,0)
    row.append(tup)
    matrix[stringy]=row
    matrix.append(row)
    print(" THE MATRIX IS:")
    print(matrix)
    return matrix

def ReturningArrayOfPicsBySplit(arrayofpicS,parts):
    arrayofpics = arrayofpicS #in order to work with local variable
    print(hashmap.keys())
    matrix = scoreOfSplits(arrayofpics, len(arrayofpics),parts) #build the matrix of scores
    print(matrix)
    matrix = pd.DataFrame(data=matrix, columns=hashmap.keys(), index=hashmap.keys())

    row,col,maxScore,maxScoreMergedImg = findMaxInMatrixForSplit(matrix, threshold=0.05) #which 2 pics has thehigh score
    w=0
    while  maxScore!= -1: # while there is a score bigger then thrseold, else -1
        w=w+1
        print("loop num: ",w)
        del1= hashmap.get(row)
        del2= hashmap.get(col)
        if row!=col: # to remove first the higher index from the array
            arraypics.remove(del1)
            arraypics.remove(del2)
            hashmap.pop(row)
            hashmap.pop(col)
        else:
            arrayofpics.remove(del1)
            hashmap.pop(row)

        matrix.drop([row], axis=1)
        matrix.drop([col], axis=1)
        matrix.drop(index=[row])
        matrix.drop(index=[col])
        mat=matrix[row][col]#adding the merged image to the original array if the score is not 0
        mat1=mat[1]
        mat2=mat[0]
        if mat2>0:
            arrayofpics.append(mat1)
            print("size of pics array: ", len(arrayofpics))
        else:
            print("    score is 0  ")

        if len(arrayofpics) > 1: # if just 1 we finished
            matrix = addColumToMatrixToSplit(matrix, arrayofpics[len(arrayofpics) - 1],arrayofpics,parts,w)
            row, col, maxScore, maxScoreMergedImg = findMaxInMatrixForSplit(matrix,threshold=0.05)  # which 2 pics has thehigh score
            print(" size of array before while: ",len(arraypics))
            print(maxScore)
        else:
            return arrayofpics

    return arrayofpics



#main:
hashmap={}
partstosplit=2

arraypics = buildingPicArray("omri",3 )#string of the name of your basee image and the amount of images you have
final_array = ReturningArrayOfPicsBySplit(arraypics,partstosplit)

print(len(final_array)) # just to check how many pics should be shown
for i in final_array:
    i.show()