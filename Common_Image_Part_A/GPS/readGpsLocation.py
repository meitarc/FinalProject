import exifread
imgArray={}
def addImag(img):
    tags = exifread.process_file(open(img, 'rb'))
    geo = {i:tags[i] for i in tags.keys() if i=='GPS GPSLatitude' or i=='GPS GPSLongitude' or i=='GPS GPSAltitude'}

    imgArray[img]=geo

addImag('0.jpg')
addImag('expImg.jpg')

print(imgArray)