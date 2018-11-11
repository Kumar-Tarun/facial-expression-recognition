#Import the OpenCV and dlib libraries
import cv2
import dlib
import os
import time
from helper import facial_landmarks, face_aligner
import imutils

#faceCascade = cv2.CascadeClassifier('/usr/local/lib/python2.7/dist-packages/cv2/data/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()

#The deisred output width and height
OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600
PATH2 = 'cohn-kanade-images/'
PATH3 = 'Faces/'
def detectAndTrackLargestFace():

    #Create two opencv named windows
    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

    cv2.moveWindow("base-image",0,100)
    cv2.moveWindow("result-image",400,100)

    cv2.startWindowThread()

    #Create the tracker we will use
    tracker = dlib.correlation_tracker()

    rectangleColor = (0,165,255)
    i = 0

    try:
    	for imagefile in sorted(os.listdir(PATH2)):
            os.mkdir(PATH3+imagefile, 0777)

            for folder in sorted(os.listdir(PATH2+imagefile+'/')):
                if(folder == '_DS_Store'):
                    continue
                os.mkdir(PATH3+imagefile+'/'+folder, 0777)
                trackingFace = 0
                
                for imagename in sorted(os.listdir(PATH2+imagefile+'/'+folder+'/')):
                    if(imagename == '_DS_Store'):
                        continue
                    temp_path = PATH3+imagefile+'/'+folder+'/'
            	    fullSizeBaseImage = cv2.imread(PATH2+imagefile+'/'+folder+'/'+imagename)
                    #fullSizeBaseImage = imutils.rotate_bound(fullSizeBaseImage, -15)
                    #fullSizeBaseImage = cv2.flip(fullSizeBaseImage, 1)
                    #Resize the image to 320x240

                    baseImage = cv2.resize( fullSizeBaseImage, ( 320, 240))

                    pressedKey = cv2.waitKey(2)
                    if pressedKey == ord('Q'):
                        cv2.destroyAllWindows()
                        exit(0)

                    resultImage = baseImage.copy()

                    if not trackingFace:

                        gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
                        faces = detector(gray, 1)
                        print("Using the cascade detector to detect face")
                        maxArea = 0
                        x = 0
                        y = 0
                        w = 0
                        h = 0
                        for face in faces:
                            if  (face.right() - face.left())*(face.bottom() - face.top()) > maxArea:
                                x = face.left()
                                y = face.top()
                                w = face.right() - x
                                h = face.bottom() - y
                                maxArea = w*h

                        if maxArea > 0 :

                            #Initialize the tracker
                            tracker.start_track(baseImage,
                                                dlib.rectangle( x-10,
                                                                y-20,
                                                                x+w+10,
                                                                y+h+20))


                            trackingFace = 1

                    if trackingFace:

                        trackingQuality = tracker.update( baseImage )

                        if trackingQuality >= 8.75:
                            tracked_position =  tracker.get_position()

                            t_x = int(tracked_position.left())
                            t_y = int(tracked_position.top())
                            t_w = int(tracked_position.width())
                            t_h = int(tracked_position.height())
                            dlib_rect = dlib.rectangle(t_x, t_y, t_x+t_w, t_y+t_h)
                            cropped_image = resultImage[t_y : t_y+t_h, t_x : t_x+t_w]
                            cv2.imwrite(temp_path+imagename, cropped_image)
                		    #cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w , t_y + t_h), rectangleColor ,1)
                            i+=1

                        else:
                            trackingFace = 0

                    largeResult = cv2.resize(resultImage,
                                             (OUTPUT_SIZE_WIDTH,OUTPUT_SIZE_HEIGHT))

                    #Finally, we want to show the images on the screen
                    cv2.imshow("base-image", baseImage)
                    cv2.imshow("result-image", largeResult)

    except KeyboardInterrupt as e:
        cv2.destroyAllWindows()
        exit(0)


if __name__ == '__main__':
    detectAndTrackLargestFace()
