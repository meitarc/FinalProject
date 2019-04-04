from SURF import funcCheck
from SURF import FetureCount
from PIL import Image
import image_slicer
from image_slicer import join
import numpy as np
import cv2
import pandas as pd

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
    for col in list(matrix):
        print(threshold)
        opo=matrix[col].max()[0]

        if opo > threshold:

            if opo > max:
                max = matrix[col].max()[0]
                img = matrix[col].max()[1]
                tuple1 = (max, img)
                str = col
    print("after loop")
    row = matrix.index[matrix[str] == tuple1][0]
    return row,str,tuple1[0],tuple1[1]

def addColumToMatrixToSplit(matrix,addedMerged,arrayofpictures,parts,counter): # not used at the moment, adding another row/colum to a matrix
    row=[]
    string="merged"
    stringy=string+str(counter)+".jpg"
    addedMerged.save(stringy)
    dict = {stringy: addedMerged}
    hashmap.update(dict)
    split1 = Split(stringy, parts)
    index = -1
    for i in arrayofpictures[:]:
        if i!=addedMerged:
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
        else:
            tuple1=(0,0)
            row.append(tuple1)

    matrix[stringy]=row[:-1]
    matrix.append(row)
    matrix.loc[stringy] = row

    return matrix

def ReturningArrayOfPicsBySplit(arrayofpicS,parts,counters,hashmap):
    arrayofpics = arrayofpicS #in order to work with local variable
    matrix = scoreOfSplits(arrayofpics, len(arrayofpics),parts) #build the matrix of scores
    matrix = pd.DataFrame(data=matrix, columns=hashmap.keys(), index=hashmap.keys())

    row,col,maxScore,maxScoreMergedImg = findMaxInMatrixForSplit(matrix, threshold=0.05) #which 2 pics has thehigh score
    w=0
    while  maxScore!= -1: # while there is a score bigger then thrseold, else -1
        w=w+1
        del1= hashmap.get(row)
        del2= hashmap.get(col)
        mat=matrix[row][col]#adding the merged image to the original array if the score is not 0
        mat1=mat[1]
        mat2=mat[0]
        if mat2>0:
            if not mat1 in arrayofpics:
                arrayofpics.append(mat1)
                print("size of pics array: ", len(arrayofpics))
        else:
            print("    score is 0  ")
        if row!=col: # if they are the same we cant remove them twice
            matrix = matrix.drop([col], axis=1)
            matrix=matrix.drop([row], axis=1)
            matrix=matrix.drop(index=[row])
            matrix=matrix.drop(index=[col])
        else:
            matrix = matrix.drop([row], axis=1)
            matrix = matrix.drop(index=[row])
        if row!=col : # to remove first the higher index from the array
            arrayofpics.remove(del1)
            arrayofpics.remove(del2)
            hashmap.pop(row)
            hashmap.pop(col)
        else:
            arrayofpics.remove(del1)
            hashmap.pop(row)
        if len(arrayofpics) > 1: # if just 1 we finished
            matrix = addColumToMatrixToSplit(matrix, arrayofpics[len(arrayofpics) - 1],arrayofpics,parts,counters)
            counters=counters+1
            row, col, maxScore, maxScoreMergedImg = findMaxInMatrixForSplit(matrix,threshold=0.05)  # which 2 pics has thehigh score
            print(" size of array before while: ",len(arraypics))
        else:
            return arrayofpics

    return arrayofpics



#main:
hashmap={}
partstosplit=4

arraypics = buildingPicArray("try",4 )#string of the name of your basee image and the amount of images you have
counters = 0
final_array = ReturningArrayOfPicsBySplit(arraypics, partstosplit, counters, hashmap)

print(len(final_array)) # just to check how many pics should be shown
for i in final_array:
    i.show()