from SURF import funcCheck
from SURF import FetureCount
from PIL import Image
import image_slicer
from image_slicer import join
import numpy as np

def buildingPicArray(string,num):
    AtoC = []
    for i in range(0,num):
        mystring = string
        mystring=mystring+str(i)+".jpg"
        ImTestA = Image.open(mystring)
        AtoC.append(ImTestA)
    arraytoReturn = []  # load array of pics
    for image in AtoC:
        arraytoReturn.append(image)
    return arraytoReturn

def CALC(img1, img2):  # that calc the % of similarity(Get 2 pics)
    y = FetureCount(img1)
    open_cv_image = np.array(img1)
    #open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    open_cv_image2 = np.array(img2)
    #open_cv_image2 = cv2.cvtColor(open_cv_image2, cv2.COLOR_BGR2GRAY)
    x = funcCheck(open_cv_image, open_cv_image2)  # move func check to flann file)
    # x = funcCheck(img1, img2)  # move func check to flann file)
    if x != 0:
        return (x / y)
    else:
        return 0

def MakeIMG(arraypic):  # that get an array of images and size and return a complete image
    image = join(arraypic)
    image.save('joined image.png')
    return image


def Split(IMG, x):  # that split and image to x pieces, return an array of images
    return image_slicer.slice(IMG, x)

def buildMatrix(array,size): #build first score matrix
    length = size-1
    k=0
    matrix=[]
    for i in range ((length-k), 0,-1):
        row=[]
        for j in range (0, (length-k)):
            if array[i]!=array[j]:
                cell=Max(CALC(array[j], array[i]),CALC(array[i],array[j]))
                tupletoadd =(i,j,cell)
                row.append(tupletoadd)

        matrix.append(row)
    return matrix



def addColumToMatrix(matrix,addedMerged): # not used at the moment, adding another row/colum to a matrix
    length = len(matrix) - 1
    row=[]
    for i in range((length - 1), 0, -1):
            cell = Max(CALC(matrix[i], addedMerged), CALC(addedMerged,matrix[i]))
            row = np.append(row,(i, matrix.index(addedMerged), cell),axis=0)
    matrix.tolist().append(row)
    return matrix


def findMaxInMatrix(matrix,threshold): #return tuple of index in matrix of highest score, the index of the 2 images in the array
    max=0
    index=(-1,-1)
    for row in matrix:
        for i in row:
            if i[2]>threshold:
                if i[2]>max:
                    max=i[2]
                    index=(i[0],i[1])
    return index

def Max(score1,score2):
    if score1>score2:
        return score1
    else:
        return score2

def splitIMG(img1,slices):
    img1.save("newBeforeSplit.jpg")
    ArrayIMG = Split('newBeforeSplit.jpg', slices)
    return ArrayIMG

def makeNewMergedIMG(arrayofImg,img2,threshold):
    ArrayIMG2 = []
    for k in arrayofImg:
            tempo=CALC(k.image, img2)
            if tempo > threshold:
                ArrayIMG2.append(k)
            else:
                print(tempo)
    if len(ArrayIMG2)>0:
        imageMerged = MakeIMG(ArrayIMG2)
        return imageMerged
    else:
        return None


def ReturningArrayOfPics(arrayofpicS):
    arrayofpics = arrayofpicS #in order to work with local variable
    matrix = buildMatrix(arrayofpics, len(arrayofpics)) #build the matrix of scores
    indexOfMax = findMaxInMatrix(matrix, threshold=0.05) #which 2 pics has thehigh score
    while indexOfMax: # while there is a score bigger then thrseold, else none
        arrayofpics[indexOfMax[0]].save("newYYY.jpg")
        splitImg = Split("newYYY.jpg", 100)
        merged = makeNewMergedIMG(splitImg, arrayofpics[indexOfMax[1]], threshold=0.05)
        if indexOfMax[0]>indexOfMax[1]: # to remove first the higher index from the array
            arrayofpics.remove(arrayofpics[indexOfMax[0]])
            arrayofpics.remove(arrayofpics[indexOfMax[1]])
        else:
            arrayofpics.remove(arrayofpics[indexOfMax[1]])
            arrayofpics.remove(arrayofpics[indexOfMax[0]])

        arrayofpics.append(merged)
        if len(arrayofpics) > 1: # if just 1 we finished
            matrix = buildMatrix(arrayofpics, len(arrayofpics))
            indexOfMax = findMaxInMatrix(matrix, threshold=0.05)
            #matrix = addColumToMatrix(matrix, arrayofpics[len(arrayofpics) - 1])
        else:
            return arrayofpics

    return arrayofpics


#main:
arraypics = buildingPicArray("dikanat",3)#string of the name of your basee image and the amount of images you have
final_array = ReturningArrayOfPics(arraypics) #returning the final common image/images
print(len(final_array)) # just to check how many pics should be shown
for i in final_array:
    i.show()

'''
AtoC = []
str0="dikanat0.jpg"
str3="dikanat1.jpg"
str2="dikanat2.jpg"
ImTest0 = Image.open(str0)
ImTestA = Image.open(str3)
ImTestB = Image.open(str2)
AtoC.append(ImTestA)
AtoC.append(ImTestB)
AtoC.append(ImTest0)
arraypics=[] # load array of pics
for image in AtoC:
    arraypics.append(image)
'''

'''
while mainIndex[0] != -1:
    AB=A[mainIndex[0]].merga(A[mainIndex[1]])
    A.remove(mainIndex[0])
    A.remove(mainIndex[1])
    A.append(AB)
    mainIndex = func(A, len(A), 5)
'''


