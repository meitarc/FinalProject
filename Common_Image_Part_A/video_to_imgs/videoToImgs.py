import cv2


def getFramesFromVideo(videoInputPath, outputFolderPath, milliSeconds):
    vidcap = cv2.VideoCapture(videoInputPath)

    success, image = vidcap.read()
    count = 1
    success = True
    while success:
        # cv2.imwrite((outputFolderPath + '/' + 'frame_%d.jpg') % count, image)
        cv2.imwrite((outputFolderPath + '/' + '%d.jpg') % count, image)
        if (milliSeconds > 0):
            vidcap.set(cv2.CAP_PROP_POS_MSEC, (milliSeconds * count))
        success, image = vidcap.read()
        count += 1
    # releasing the video file after we're done to free memmory space.
    vidcap.release()
# return


#main
for i in range(1, 10):
    getFramesFromVideo("input/"+str(i)+".MP4",str(i)+"/",500)
