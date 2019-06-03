import exifread
imgArray={}
sameGpsPics=[]
def first(img):
    tags = exifread.process_file(open(img, 'rb'))
    geo = {i:tags[i] for i in tags.keys() if i=='GPS GPSLatitude' or i=='GPS GPSLongitude' or i=='GPS GPSAltitude'}
    imgArray["first"]=geo

def addImag(img):
    tags = exifread.process_file(open(img, 'rb'))
    geo = {i:tags[i] for i in tags.keys() if i=='GPS GPSLatitude' or i=='GPS GPSLongitude' or i=='GPS GPSAltitude'}
    imgArray[img]=geo

first('pics_for_gps/20190603_172000.jpg')

import os
for filename in os.listdir('pics_for_gps'):
    addImag("pics_for_gps/"+filename)

for key, value in zip(imgArray,imgArray.values()):
    #print(imgArray[key].values())
    #print(imgArray["first"].values())
    if str(imgArray[key].values()) == str(imgArray["first"].values()):
        sameGpsPics.append(key)
        print("true")
    else:
        print("false")

print(sameGpsPics)