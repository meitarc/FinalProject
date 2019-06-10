#from SURF import *
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
# from sklearn.datasets.samples_generator import make_blobs
# from sklearn.preprocessing import StandardScaler
import cv2
import pandas as pd
from functions import*


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

def extractKeyPt1(kp1, des):
    dict_Kp_pt = {}
    counter=0
    for kp, ds in zip(kp1, des):
        if kp.pt in dict_Kp_pt.keys():
            val=dict_Kp_pt.get(kp.pt)
            importance=val[2]
            if importance>kp.response:
                dict_Kp_pt.update({kp.pt: (kp.size, kp.angle, kp.response, kp.octave, kp.class_id, ds)})
            else:
                continue
        else:
            dict_Kp_pt.update({kp.pt: (kp.size, kp.angle, kp.response, kp.octave, kp.class_id, ds)})
    return dict_Kp_pt

def top_of_kp__from_clusters(image1):
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = np.array(image1)
    kp, des = sift.detectAndCompute(img1, None)
    #print("kp len ",len(kp))
    dict_Kp_pt1 = extractKeyPt1(kp, des)
    #print("length of dict: ",len(dict_Kp_pt1.keys()))
    cord_array = DB_SCAN(kp)
    clusters = pd.DataFrame(data=cord_array)
    clusters = clusters.transpose()
    # clusters.to_csv('clusters.csv',index=False)
    dfcounter = 0
    list = []
    counter_else = 0
    counter_yes=0
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
            if k is not None:
                if k in dict_Kp_pt1.keys():
                    values=dict_Kp_pt1.get(k)
                    counter_yes = counter_yes + 1
                    # val = dict_Kp_pt1.get(k)
                    importance = values[2]
                    descriptor = values[5]
                    tuple1 = (k[0], k[1], values[0], values[1], values[2], values[3], values[4])
                    arr.append(tuple1)
                    arrImp.append(importance)
                    arrDesc.append(descriptor)
                    '''
                    values = dict_Kp_pt1.get(k)
                    if isinstance(values,tuple):
                            #print("should be tuple : ",type(values))
                            counter_yes=counter_yes+1
                            #val = dict_Kp_pt1.get(k)
                            importance = values[3]
                            descriptor= values[6]
                            tuple1 = (k[0], k[1], values[1], values[2], values[3], values[4], values[5])
                            arr.append(tuple1)
                            arrImp.append(importance)
                            arrDesc.append(descriptor)
                    else:
                        #print(type(values))
                        #print("should be list : ", type(values))
                        counterbigi=0
                        for value in values:
                            #print(type(value), value)
                            #print("should be tuple 2 : ",type(value))
                            #print(counterbigi)
                            counterbigi=counterbigi+1
                            counter_yes = counter_yes + 1
                            # val = dict_Kp_pt1.get(k)
                            #print(value[2])
                            importance = value[2]
                            descriptor = value[5]
                            tuple1 = (k[0], k[1], value[0], value[1], value[2], value[3], value[4])
                            arr.append(tuple1)
                            arrImp.append(importance)
                            arrDesc.append(descriptor)
                    '''
                else:
                    counter_else = counter_else+  1
                # tuple1 = -1
                # importance=-1
                # arr.append(tuple1)
                # arrImp.append(importance)

        dataframe = pd.DataFrame(data=[arr, arrImp,arrDesc])
        dataframe = dataframe.transpose()
        dataframe.columns = listnames
        list.append(dataframe)
        dfcounter = dfcounter + 1
    #print("yes is ", counter_yes)
    #print("else is ", counter_else)
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

    return list2,list

def tuple_to_kp(t):
    #tapleKP = (t[0], t[1], t[2], t[3], t[4], t[5], t[6])
    kp = cv2.KeyPoint(t[0], t[1], t[2], t[3], t[4], int(t[5]), int(t[6]))
    return kp

#or list of DF's
def prepareDataforList(somelist):
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

#for one DF
def prepareDataforDF(somelist,counter):
    #desc_array = np.array([])
    desc_array = []
    kp_list=[]
    string1='cluster'
    string2='descriptor'
    string3=string1+str(counter)+string2
    string5=string1+str(counter)
    counter=counter+1
    #desc_array = np.append(desc_array, i[string3])
    for h in somelist[string3]:
        desc_array.append(h)

    for j in somelist[string5]:
        kp_list.append(tuple_to_kp(j))
    return kp_list,desc_array


def clusterSize(clusterDF,clusterNumber):
    string="cluster"
    maxX=0
    maxY=0
    minX=clusterDF[string+str(clusterNumber)][0][0]
    minY=clusterDF[string+str(clusterNumber)][0][1]
    for x in clusterDF[string+str(clusterNumber)]:
        if x[0]>maxX:
            maxX=x[0]
        if x[0]<minX:
            minX=x[0]
    for y in clusterDF[string+str(clusterNumber)]:
        if y[1]>maxY:
            maxY=y[1]
        if y[1]<minY:
            minY=y[1]
    return minY,maxY,minX,maxX
# main

# regular func check getting 2 images

imagefromserver = cv2.imread('try1.jpg')
imagefromclient = cv2.imread('try0.jpg')

'''
in server:
func check of good clusters, in server
top list is best 20 % of each cluster
all list is just all of the lists combined
'''
top_list,all_lists = top_of_kp__from_clusters(imagefromserver)
kpOne,desOne=prepareDataforList(top_list)# kps of to

'''
in client:
'''
#x2,kp2=clientFuncCheck(kpOne, desOne, imagefromclient)

list_of_size=[]
counter=0
for i in all_lists:
    minY, maxY, minX, maxX=clusterSize(i,counter)
    #print(minY,maxY,minX,maxX)
    size,center=SizeandCenter(minY,maxY,minX,maxX)
    tuple_of_sizes=(minY,maxY,minX,maxX,size,center)
    list_of_size.append(tuple_of_sizes)
    counter=counter+1

counter=0
sumX=0
sumPer =0
arr=[]
#search for each in list if is a good cluster:
for i in top_list:
    kpOne, desOne = prepareDataforDF(i,counter)
    matches, kp1 = clientFuncCheck(kpOne, desOne, imagefromclient)
    sumX=sumX+matches
    sumPer=sumPer+len(kpOne)
    print("x2 is :", matches,counter)
    if len(kpOne) != 0:
        print("percentege is: ", matches / len(kpOne),counter)
    if len(kpOne)!=0:
        if matches / len(kpOne) < 0.3:
            arr.append(counter)
            size=list_of_size[counter][4]
            center=list_of_size[counter][5]
            #print( list_of_size[counter][0])
            crop_img = imagefromclient[ int(list_of_size[counter][0]):int(list_of_size[counter][1]) , int(list_of_size[counter][2]):int(list_of_size[counter][3])]
            cv2.imwrite('cropped'+str(counter)+'.jpg',crop_img)
    counter=counter+1

print()
print("good clusters : ",arr)
print("sum ",sumX)
print("sum per ",sumPer)
print("total ",sumX/sumPer)


#show features on image 1 from ALL of clusters
kpOne,desOne=prepareDataforList(all_lists)

from matplotlib import pyplot as plt

image=cv2.imread('try1.jpg')
img = np.array(image)
for i in kpOne:
    #print(type(i))
    cv2.circle(img, (int(i.pt[0]), int(i.pt[1])), 10, (255, 0, 255), -1)

plt.imshow(img,),plt.show()

#show features on image 1 from top of good clusters
kpOne,desOne=prepareDataforList(top_list)

from matplotlib import pyplot as plt

image=cv2.imread('try1.jpg')
img = np.array(image)
for i in kpOne:
    #print(type(i))
    cv2.circle(img, (int(i.pt[0]), int(i.pt[1])), 10, (255, 0, 255), -1)

plt.imshow(img,),plt.show()


string='cluster'
list3=[]
print(arr)

#SHOW and list3 will be kps of all the clusters in good clusters
for j in arr:
    string2=string+str(j)
    for i in all_lists[j][string2]:
        list3.append(i)
print("good")

array=[]
for y in list3:
    array.append(tuple_to_kp(y))


from matplotlib import pyplot as plt

image=cv2.imread('try1.jpg')
img = np.array(image)
for i in array:
    #print(type(i))
    cv2.circle(img, (int(i.pt[0]), int(i.pt[1])), 10, (255, 0, 255), -1)


plt.imshow(img,),plt.show()

#show list3 top of good
list3=[]
for j in arr:
    string2=string+str(j)
    for i in top_list[j][string2]:
        list3.append(i)


array=[]
for y in list3:
    array.append(tuple_to_kp(y))


from matplotlib import pyplot as plt

image=cv2.imread('try1.jpg')
img = np.array(image)
for i in array:
    #print(type(i))
    cv2.circle(img, (int(i.pt[0]), int(i.pt[1])), 10, (255, 0, 255), -1)


plt.imshow(img,),plt.show()

'all'
'top of all'
'all of good'
'top of good'


