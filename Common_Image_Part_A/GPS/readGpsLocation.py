def find_imgs_same_gps(imgSource, folderPath):
    import exifread
    firstImg={}
    imgArray={}
    sameGpsPics=[]
    def first(img):
        tags = exifread.process_file(open(img, 'rb'))
        geo = {i:tags[i] for i in tags.keys() if i=='GPS GPSLatitude' or i=='GPS GPSLongitude' or i=='GPS GPSAltitude'}
        firstImg["first"]=geo

    def addImag(img):
        tags = exifread.process_file(open(img, 'rb'))
        geo = {i:tags[i] for i in tags.keys() if i=='GPS GPSLatitude' or i=='GPS GPSLongitude' or i=='GPS GPSAltitude'}
        imgArray[img]=geo

    first(imgSource)

    import os
    for filename in os.listdir(folderPath):
        addImag(folderPath+"/"+filename)

    for key, value in zip(imgArray,imgArray.values()):
        if str(imgArray[key].values()) == str(firstImg["first"].values()):
            sameGpsPics.append(key)

    return sameGpsPics


myImgArray=find_imgs_same_gps("pics_for_gps/20190603_172000.jpg","pics_for_gps")
print(myImgArray)
print(len(myImgArray))