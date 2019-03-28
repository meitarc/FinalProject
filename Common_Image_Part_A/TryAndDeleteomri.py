from SURF import funcCheck
from SURF import FetureCount
from PIL import Image
import image_slicer
from image_slicer import join
import numpy as np
import cv2

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
    open_cv_image2 = np.array(img2)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    open_cv_image2 = cv2.cvtColor(open_cv_image2, cv2.COLOR_BGR2GRAY)
    #open_cv_image = np.array(img1)
    #open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    #open_cv_image2 = np.array(img2)
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

def scoreOfSplits(array,sizeofarray,parts):
    length = sizeofarray-1
    k=0
    matrix=[]
    for i in range((length-k), 0, -1):
        row =[]
        print("another row ", i)
        for j in range(0, (length-k)):
            print("another col ", j)
            if i != j or array[i]!=array[j]:
                array[i].save("newSCORE.jpg")
                splitImg = Split("newSCORE.jpg", parts)
                merged, counter_of_good = makeNewMergedIMG(splitImg, array[j], threshold=0.05)
                array[j].save("newSCORE2.jpg")
                splitImg2 = Split("newSCORE2.jpg", parts)
                merged2, counter_of_good2 = makeNewMergedIMG(splitImg2, array[i], threshold=0.05)
                if(counter_of_good>counter_of_good2):
                    tupletoadd =(i,j,counter_of_good,merged)
                else:
                    tupletoadd =(i,j,counter_of_good2,merged2)
                row.append(tupletoadd)
            else:
                tupletoadd=(i,j,0,array[i])
                row.append(tupletoadd)
        matrix.append(row)
    print("type of just built matrix", type(matrix))
    print(matrix)
    return matrix


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

def addColumToMatrix(matrix,addedMerged): # not used at the moment, adding another row/colum to a matrix
    length = len(matrix) - 1
    row=[]
    for i in range((length - 1), 0, -1):
            cell = Max(CALC(matrix[i], addedMerged), CALC(addedMerged,matrix[i]))
            row = np.append(row,(i, matrix.index(addedMerged), cell),axis=0)
    matrix.tolist().append(row)
    return matrix


def addColumToMatrixToSplit(matrix,addedMerged,arrayofpictures,parts): # not used at the moment, adding another row/colum to a matrix
    print("adding")
    #length = len(matrix)
    row=[]
    addedMerged.save("merged.jpg")
    split1 = Split("merged.jpg", parts)
    index = -1
    for i in arrayofpictures[:-1]:
        index=index+1
        i.save("i.jpg")
        split2 = Split("i.jpg", parts)
        img1,score1=makeNewMergedIMG(split1, i,threshold=0.05)
        img2,score2=makeNewMergedIMG(split2,addedMerged,threshold=0.05)
        if score1>score2:
            tuple1 = (len(arrayofpictures)-1,index,score1,img1)
        else:
            tuple1 = (len(arrayofpictures)-1,index,score2,img2)
        row.append(tuple1)
    matrix.append(row)

    return matrix
        #cell = Max(CALC(matrix[i], addedMerged), CALC(addedMerged,matrix[i]))
        #cell,img= makeNewMergedIMG(matrix[i],addedMerged)
        #row = np.append(row,(i, matrix.index(addedMerged), cell,img),axis=0)
    #matrix.append(row)
    #return matrix


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

def findMaxInMatrixForSplit(matrix,threshold): #return tuple of index in matrix of highest score, the index of the 2 images in the array
    max=0
    index=(-1,-1)
    counterRow=-1
    counterRowtoreturn=-1
    countercoltoreturn=-1
    for row in matrix:
        counterRow = counterRow + 1
        countrCol = -1
        for i in row:
            countrCol = countrCol + 1
            if i[2]>threshold:
                if i[2]>max:
                    max=i[2]
                    index=(i[0],i[1])
                    countercoltoreturn=countrCol
                    counterRowtoreturn=counterRow
    print("type of 1 matrix", type(matrix))

    return index,counterRowtoreturn,countercoltoreturn

def Max(score1,score2):
    if score1>score2:
        return score1
    else:
        return score2
'''
def splitIMG(img1,slices):
    img1.save("newBeforeSplit.jpg")
    ArrayIMG = Split('newBeforeSplit.jpg', slices)
    return ArrayIMG
'''




def ReturningArrayOfPics(arrayofpicS,parts):
    arrayofpics = arrayofpicS #in order to work with local variable
    matrix = buildMatrix(arrayofpics, len(arrayofpics)) #build the matrix of scores
    indexOfMax = findMaxInMatrix(matrix, threshold=0.05) #which 2 pics has thehigh score
    while indexOfMax[0]!= -1: # while there is a score bigger then thrseold, else -1
        arrayofpics[indexOfMax[0]].save("newYYY.jpg")
        splitImg = Split("newYYY.jpg", parts)
        merged, counter_of_good = makeNewMergedIMG(splitImg, arrayofpics[indexOfMax[1]], threshold=0.05)
        if indexOfMax[0]>indexOfMax[1]: # to remove first the higher index from the array
            arrayofpics.remove(arrayofpics[indexOfMax[0]])
            arrayofpics.remove(arrayofpics[indexOfMax[1]])
        else:
            arrayofpics.remove(arrayofpics[indexOfMax[1]])
            arrayofpics.remove(arrayofpics[indexOfMax[0]])

        arrayofpics.append(merged)
        if len(arrayofpics) > 1: # if just 1 we finished
            matrix = buildMatrix(arrayofpics, len(arrayofpics))#new score matrix
            indexOfMax = findMaxInMatrix(matrix, threshold=0.05)
            #matrix = addColumToMatrix(matrix, arrayofpics[len(arrayofpics) - 1])
        else:
            return arrayofpics

    return arrayofpics


def ReturningArrayOfPicsBySplit(arrayofpicS,parts):
    arrayofpics = arrayofpicS #in order to work with local variable
    #matrix = buildMatrix(arrayofpics, len(arrayofpics)) #build the matrix of scores
    matrix = scoreOfSplits(arrayofpics, len(arrayofpics),parts) #build the matrix of scores
    indexOfMax,row,col = findMaxInMatrixForSplit(matrix, threshold=0.05) #which 2 pics has thehigh score
    w=0
    while indexOfMax[0]!= -1: # while there is a score bigger then thrseold, else -1
        w=w+1
        print("w : ",w)
        print(len(matrix))
        print(len(matrix[0]))
        print("row:",row)
        print("col:", col)
        print("size of pics array: ",len(arrayofpics))
        print("to remove: ",indexOfMax[0],indexOfMax[1])
        if indexOfMax[0]>indexOfMax[1]: # to remove first the higher index from the array
            arrayofpics.remove(arrayofpics[indexOfMax[0]])
            arrayofpics.remove(arrayofpics[indexOfMax[1]])
        else:
            if indexOfMax[0]<indexOfMax[1]:
                arrayofpics.remove(arrayofpics[indexOfMax[1]])
                arrayofpics.remove(arrayofpics[indexOfMax[0]])
            else:
                arrayofpics.remove(arrayofpics[indexOfMax[0]])
        print("type of 2 matrix", type(matrix))

        print("size of pics array: ",len(arrayofpics))
        print("len of row: ",len(matrix))
        print("len of col: ",len(matrix[0]))
        mat=matrix[row][col]#adding the merged image to the original array if the score is not 0
        mat1 = mat[3]
        mat2=mat[2]
        print("type of 3 matrix", type(matrix))

        if mat2>0:
            arrayofpics.append(mat1)
            print("size of pics array: ", len(arrayofpics))
        else:
            print("    score is 0  ")
        #matrix[row][col]=(row,col,0,0)
        print("PASSS")
        print("before remove matrix:")
        print("index to remove: ",indexOfMax[0], indexOfMax[1])
        print("type of 4 matrix", type(matrix))

        for i in matrix:
            print(i)
        #for value in matrix[:]:
            #value1 = [e for e in value if (e[0] != indexOfMax[0]) or (e[0] != indexOfMax[1]) or (e[1] != indexOfMax[0]) or (e[1] != indexOfMax[1])]
            #matrix[value]=value1
        del matrix[row]
        print("after del row")
        for i in matrix:
            print(i)
        for row in matrix:
            del row[col]

            #for e in value:
                #print("e:", e)
                #matrix = matrix.remove(value[e])


        #matrix = np.delete(matrix, indexOfMax[0], axis=0)
        #matrix = np.delete(matrix, indexOfMax[1], axis=1)
        print("after remove: ")
        for i in matrix:
            print(i)
        if len(arrayofpics) > 1: # if just 1 we finished
            matrix = addColumToMatrixToSplit(matrix, arrayofpics[len(arrayofpics) - 1],arrayofpics,parts)
            indexOfMax,row,col = findMaxInMatrixForSplit(matrix, threshold=0.05)
            print(" size of array before while: ",len(arraypics))
            print(indexOfMax[0])
        else:
            return arrayofpics

    return arrayofpics


#main:
partstosplit=16

arraypics = buildingPicArray("dikanat",5)#string of the name of your basee image and the amount of images you have
#matrix = scoreOfSplits(arraypics, len(arraypics), partstosplit)  # build the matrix of scores
#for i in arraypics:
#    print(i)
#final_array = ReturningArrayOfPics(arraypics,partstosplit) #returning the final common image/images
final_array = ReturningArrayOfPicsBySplit(arraypics,partstosplit)
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


