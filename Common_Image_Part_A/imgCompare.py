from SURF import funcCheck
from SURF import FetureCount
from PIL import Image
import image_slicer
from image_slicer import join
import numpy as np
import cv2
import pandas as pd
import compare
from io import BytesIO


def buildingPicArray(string, num, typeimg):
    AtoC = []
    for i in range(0, num):
        mystring = string
        mystring = mystring + str(i) + "." + typeimg
        ImTestA = Image.open(mystring)
        AtoC.append(ImTestA)
        hashmap.update({mystring: ImTestA})
    # print(hashmap)
    # print(AtoC)
    #
    # for i in range(len(AtoC) - 1):
    #    if (AtoC[i].size != AtoC[i + 1].size):
    #        AtoC[i + 1] = AtoC[i + 1].resize(AtoC[i].size, Image.ANTIALIAS)
    #        b = BytesIO()
    #        AtoC[i + 1].save(b, format="jpeg")
    #        AtoC[i + 1] = Image.open(b)
    #        hashmap.update({string+str(i+1)+"."+typeimg: AtoC[i+1]})
    # print(AtoC)

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


def scoreOfSplits(array, sizeofarray, parts, hashmap, bBlocks):
    length = sizeofarray
    black = Image.open("Black_Image.jpg")
    matrix = []
    for i in range(0, length):
        row = []
        print("another row ", i)
        for j in range(0, (length)):
            print("another col ", j)
            key_list = list(hashmap.keys())
            val_list = list(hashmap.values())
            name1 = key_list[val_list.index(array[i])]
            name2 = key_list[val_list.index(array[j])]

            # for (key, val) in hashmap.items():
            #    if val == array[i]:
            #        name1 = key
            # for (key, val) in hashmap.items():
            #    if val == array[j]:
            #        name2 = key
            # name1 = [key for (key, value) in hashmap.items() if value == array[i]]
            # name2 = [key for (key, value) in hashmap.items() if value == array[j]]
            # name1=hashmap.keys()[hashmap.values().index(array[i])]
            # name2=hashmap.keys()[hashmap.values().index(array[j])]
            name3 = name1 + name2
            if i != j or array[i] != array[j]:
                new_width, new_height = array[i].size
                name1 = key_list[val_list.index(array[j])]
                array[j] = array[j].resize((new_width, new_height), Image.ANTIALIAS)
                hashmap.pop(name1)
                hashmap.update({name1: array[j]})
                array[i].save("newSCORE.jpg")
                splitImg = Split("newSCORE.jpg", parts)
                merged, counter_of_good, name3 = makeNewMergedIMG(splitImg, name1, array[j], name2, parts,
                                                                  threshold=0.05, bBlocks=bBlocks)
                array[j].save("newSCORE2.jpg")
                splitImg2 = Split("newSCORE2.jpg", parts)
                merged2, counter_of_good2, name3 = makeNewMergedIMG(splitImg2, name2, array[i], name1, parts,
                                                                    threshold=0.05, bBlocks=bBlocks)
                if (counter_of_good > counter_of_good2):
                    tupletoadd = (counter_of_good, merged, name3)
                else:
                    tupletoadd = (counter_of_good2, merged2, name3)
                row.append(tupletoadd)
            else:
                tupletoadd = (0, black, name3)
                row.append(tupletoadd)
        matrix.append(row)
    return matrix


'''

def makeNewMergedIMG(arrayofImg,name1,img2,name2,threshold):
    ArrayIMG2 = []
    countOfGood =0
    name=name1+name2
    black = Image.open("Black_Image.jpg")
    for k in arrayofImg:
            tempo=CALC(k.image, img2)
            if tempo > threshold:
                ArrayIMG2.append(k)
                countOfGood=countOfGood+1
            #else:
                #print("tempo is: ",tempo)
    if len(ArrayIMG2)>0:
        imageMerged = MakeIMG(ArrayIMG2)
        return imageMerged,countOfGood,name
    else:
        return black,0,name

'''


def makeNewMergedIMG(arrayofImg, name1, img2, name2, parts, threshold, bBlocks):  # need to add parts to function call
    ArrayIMG2 = []
    countOfGood = 0
    name = name1 + name2
    black = Image.open("Black_Image.jpg")

    img2.save("newSCORE.jpg")
    splitImg = Split("newSCORE.jpg", parts)

    for k, l in zip(arrayofImg, splitImg):
        tempo = CALC(k.image, img2)
        if tempo > threshold:
            ArrayIMG2.append(k)
            countOfGood = countOfGood + 1
            print("if 1 true")
        else:
            #print("tempo is: ", tempo)
            #print("size1: "+str(l.image.size)+" size2: "+str(k.image.size))
            if(FetureCount(k.image)<100):
                print("fetureCount<100")
                if (l.image.size != k.image.size):
                    print("tiles size different")
                    l.image = l.image.resize(k.image.size, Image.ANTIALIAS)
                    b = BytesIO()
                    l.image.save(b, format="jpeg")
                    l.image = Image.open(b)
                k.image.save("splitedImg1.jpg")
                l.image.save("splitedImg2.jpg")
                tempo2 = compare.main_compare("splitedImg1.jpg", "splitedImg2.jpg")
                if (tempo2 > 0.7):
                    ArrayIMG2.append(k)
                    countOfGood = countOfGood + 1
                    print("tempo2>0.7: ",tempo2)
                else:
                    ArrayIMG2.append(bBlocks[0])
                    print("add black - tempo2<0.7: ", tempo2)
            else:
                ArrayIMG2.append(bBlocks[0])
                print("add black - fetureCount>100 ")

    if len(ArrayIMG2) > 0:
        imageMerged = MakeIMG(ArrayIMG2)
        return imageMerged, countOfGood, name
    else:
        return black, 0, name


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
                name = e[2]
    return max, img, name


def findMaxInMatrixForSplit(matrix,
                            threshold):  # return tuple of index in matrix of highest score, the index of the 2 images in the array
    black = Image.open("Black_Image.jpg")
    img1 = black
    max = 0
    tuple1 = (0, 0, 0)
    tuple2 = (0, 0, 0)
    name1 = "sss"
    for col in list(matrix):
        max1, img, name = find_max_in_column(matrix, col, max, threshold=0.05)
        print("max1 is : ", max1, "img1 ", img1, "name ", name)
        if max1 > float(threshold):
            if max1 > float(max):
                max = max1
                img1 = img
                name1 = name
                tuple2 = (max, img1, name1)
                str = col
            else:
                tuple1 = (0, img1, name1)
        else:
            tuple1 = (0, img1, name1)
            if tuple1 is not None:
                print("tuple1is ", tuple1, "of col ", col)
    if tuple2[0] == 0:
        return col, col, -1, black, name1
    else:
        row = matrix.index[matrix[str] == tuple2][0]
        # row, col, maxScore, maxScoreMergedImg
        return row, str, tuple2[0], tuple2[1], tuple2[2]


def addColumToMatrixToSplit(matrix, addedMerged, arrayofpictures, parts, counter,
                            bBlocks):  # not used at the moment, adding another row/colum to a matrix
    row = []
    string = "merged"
    black = Image.open("Black_Image.jpg")
    stringy = string + str(counter) + ".jpg"
    addedMerged.save(stringy)
    dict = {stringy: addedMerged}
    hashmap.update(dict)
    split1 = Split(stringy, parts)
    index = -1
    # name1 = hashmap.keys()[hashmap.values().index(arrayofpictures[i])]
    # for (key, val) in hashmap.items():
    #    if val == addedMerged:
    #        name1=key
    key_list = list(hashmap.keys())
    val_list = list(hashmap.values())
    name1 = key_list[val_list.index(addedMerged)]

    # name2 = key_list[val_list.index(array[j])]
    # name1 = [key for (key, value) in hashmap.items() if value == addedMerged]

    for i in arrayofpictures[:]:
        if i != addedMerged:
            new_width, new_height = addedMerged.size
            key_list = list(hashmap.keys())
            val_list = list(hashmap.values())
            name1 = key_list[val_list.index(i)]
            i = i.resize((new_width, new_height), Image.ANTIALIAS)
            hashmap.pop(name1)
            hashmap.update({name1: i})
            index = index + 1
            i.save("i.jpg")
            split2 = Split("i.jpg", parts)
            # name = hashmap.keys()[hashmap.values().index(arrayofpictures[i])]
            # for (key,val) in hashmap.items():
            #    if val == arrayofpictures[i]:
            #        name=key
            key_list = list(hashmap.keys())
            val_list = list(hashmap.values())
            print(i)
            print(val_list.index(i))
            print(key_list[val_list.index(i)])
            name = key_list[val_list.index(i)]
            # name = key_list.index(val_list.index(arrayofpictures[i]))
            # name = key_list[val_list.index(arrayofpictures[i])]

            img1, score1, name1 = makeNewMergedIMG(split1, stringy, i, name1, parts, threshold=0.05, bBlocks=bBlocks)
            img2, score2, name2 = makeNewMergedIMG(split2, name, addedMerged, stringy, parts, threshold=0.05,
                                                   bBlocks=bBlocks)
            if score1 > score2:
                tuple1 = (score1, img1, name1)
            else:
                tuple1 = (score2, img2, name2)
            row.append(tuple1)
        else:
            tuple1 = (0, black, name1)
            row.append(tuple1)

    matrix[stringy] = row[:-1]
    matrix.append(row)
    matrix.loc[stringy] = row

    return matrix


def ReturningArrayOfPicsBySplit(matrix, arrayofpicS, parts, counters, hashmap, bBlocks):
    row, col, maxScore, maxScoreMergedImg, name = findMaxInMatrixForSplit(matrix,
                                                                          threshold=0.05)  # which 2 pics has thehigh score
    w = 0
    while maxScore != -1 or len(arrayofpicS) > 1:  # while there is a score bigger then thrseold, else -1
        w = w + 1
        # mat = matrix[row][col]  # adding the merged image to the original array if the score is not 0
        print("round number : ", w)
        print("mat2 is ", maxScore)
        # mat1 = mat[1]
        print("mat 1 is ", maxScoreMergedImg)
        # mat2 = mat[0]
        print("mat name is ", name)

        if maxScore > 0:
            if not maxScoreMergedImg in arrayofpicS:
                hashmap.update({name: maxScoreMergedImg})
                arrayofpicS.append(maxScoreMergedImg)
                print("size of pics array: ", len(arrayofpicS))
        else:
            print("    score is 0  ")
            return arrayofpicS
        del1 = hashmap.get(row)
        del2 = hashmap.get(col)

        if row != col:  # if they are the same we cant remove them twice
            matrix = matrix.drop([col], axis=1)
            matrix = matrix.drop([row], axis=1)
            matrix = matrix.drop(index=[row])
            matrix = matrix.drop(index=[col])
        else:
            matrix = matrix.drop([row], axis=1)
            matrix = matrix.drop(index=[row])
        if row != col:  # to remove first the higher index from the array
            arrayofpicS.remove(del1)
            arrayofpicS.remove(del2)
            hashmap.pop(row)
            hashmap.pop(col)
        else:
            arrayofpicS.remove(del1)
            hashmap.pop(row)
        if len(arrayofpicS) > 1:  # if just 1 we finished
            matrix = addColumToMatrixToSplit(matrix, arrayofpicS[len(arrayofpicS) - 1], arrayofpicS, parts, counters,
                                             bBlocks)
            counters = counters + 1
            row, col, maxScore, maxScoreMergedImg, name = findMaxInMatrixForSplit(matrix,
                                                                                  threshold=0.05)  # which 2 pics has thehigh score
            print(" size of array before while: ", len(arraypics))
        else:
            return arrayofpicS

    return arrayofpicS


# Create Black Image
def CreateBlackBlock(picArray, numPices):
    imBlack = Image.new('RGB', picArray[0].size, (0, 0, 0))
    imBlack.save("Black_Image.jpg")
    blackTiles = (image_slicer.slice("Black_Image.jpg", numPices, False))
    print(blackTiles[0].coords)
    return blackTiles


# Print Black Dictionary key and value
def PrintBlackDictionary(blackD):
    print("Print Black Dictionary \n")
    for key, value in blackD.items():
        print("{},{} ".format(key, value))


# main:
hashmap = {}
partstosplit = 4
arraypics = buildingPicArray("try", 2, "jpg")  # string of the name of your basee image and the amount of images you have
print(arraypics)
bBlocks = CreateBlackBlock(arraypics, partstosplit)
matrix = scoreOfSplits(arraypics, len(arraypics), partstosplit, hashmap, bBlocks)
matrix = pd.DataFrame(data=matrix, columns=hashmap.keys(), index=hashmap.keys())

counters = 0
final_array = ReturningArrayOfPicsBySplit(matrix, arraypics, partstosplit, counters, hashmap, bBlocks)

print(len(final_array))  # just to check how many pics should be shown
for i in final_array:
    i.show()
