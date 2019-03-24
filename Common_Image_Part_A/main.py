from SURF import funcCheck
from SURF import FetureCount
from PIL import Image
import image_slicer
from image_slicer import join
import numpy as np
import cv2
import compare
def  CALC(img1,img2):#that calc the % of similarity(Get 2 pics)
        y=FetureCount(img1)
        print("number of img1 features: ",y)
        open_cv_image = np.array(img1)
        open_cv_image=cv2.cvtColor(open_cv_image,cv2.COLOR_BGR2GRAY)
        open_cv_image2 = np.array(img2)
        open_cv_image2 = cv2.cvtColor(open_cv_image2, cv2.COLOR_BGR2GRAY)
        x= funcCheck(open_cv_image,open_cv_image2)#move func check to flann file)
        # x = funcCheck(img1, img2)  # move func check to flann file)
        print("similar features: ", x)
        if x != 0:
            return(x / y)
        else:
            return 0

def MakeIMG(arraypic):# that get an array of images and size and return a complete image
        image = join(arraypic)
        image.save('joined image.png')
        return image


def Split(IMG,x):# that split and image to x pieces, return an array of images
    return image_slicer.slice(IMG, x)

    '''
    import cv2
    #img = cv2.imread('image.png')
    img=IMG
    for r in range(0, img.shape[0], 30):
        for c in range(0, img.shape[1], 30):
            cv2.imwrite(f"img{r}_{c}.png", img[r:r + 30, c:c + 30, :])
    '''
# MAIN

threshold = 1
AtoC = []
# build array of images
num = 3 # number of images
#names=[]

#for k in range(num):
    #print(k)
    #str1 = "school"
    #str2 = str(k)
    #str1 = str1+str2+".jpeg"
str0="1.jpg"
#str3="img2.jpeg"
#str1="school4.jpeg"
str2="2.jpg"
compare.main_compare(str0,str2)
    #names.append(str1)
ImTest0 = Image.open(str0)
#ImTestA = Image.open(str3)
ImTestB = Image.open(str2)


#AtoC.append(ImTestA)
AtoC.append(ImTestB)
AtoC.append(ImTest0)




arrayofpics=[] # load array of pics
for image in AtoC:
    arrayofpics.append(image)

# when we will need score matrix:
# ScoreMatrix = [[0 for y in range(4)] for x in range(4)]
while len(arrayofpics)>1:
    print(len(arrayofpics))
    max =0
    index=(0,0)
    i=1
    counter1 = 0
    for i in arrayofpics:
        counter1 = counter1+1
        counter2 = 0
        for j in arrayofpics:
            counter2 = counter2+1
            if i != j :  # don't compare images to itself
                print("starting Calc ")
                temp = CALC(i, j)
                print("needs to be 178 : ",temp)
                if temp > max:
                      max = temp
                      index = (i, j)
     #                 realcounterI=counter1
    #                  realcounterJ=counter2

    #spliter=spliter+1
    n = 32 # numberOfParts
    index[0].save("YYY.jpg")
    ArrayIMG = Split('YYY.jpg', n)
    # ArrayIMG = Split(names[realcounterI], n)
    ArrayIMG2 = []
    for k in ArrayIMG:
            tempo=CALC(k.image, index[1])
            #if( not tempo):
            #print(tempo)
            if tempo > 0.05:
                #print(tempo)
                ArrayIMG2.append(k)
            else: print(tempo)
    #print(ArrayIMG2)
    if len(ArrayIMG2)>0:
        IMAGE=MakeIMG(ArrayIMG2) # maybe will chane from join to new image
        #if (not arrayofpics.index(index[0])):
        arrayofpics.remove(index[0])
        arrayofpics.remove(index[1])
            #names.remove(names[realcounterI])
            #names.remove(names[realcounterJ])
        print("creating")
        IMAGE.save("C:\\Users\\Meitar\\קבצי לימודים\\שנה ג\\פרויקט גמר\\FinalProject\\Common_Image_Part_A\output\\XXe.jpg")
            #names.append("XXX.jpg")
        arrayofpics.append(IMAGE)
        print(len(arrayofpics))
    else:
        #bool=False
        print("problem")




