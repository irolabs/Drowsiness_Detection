import cv2 as cv
import dlib
from  scipy.spatial import distance

#EAR = Eye Aspect Ratio
earValue = 0.26
topText = "Drowsy Detected"
windowName = "Drowsiness Detection"

#Function for Calculated Eye Aspect Ratio
def calculatedEAR(eye):
    A = distance.euclidean(eye[1],eye[5])
    B = distance.euclidean(eye[2],eye[4])
    C = distance.euclidean(eye[0],eye[3])
    eyeAspectRatio = (A+B)/(2.0*C)
    return eyeAspectRatio
#Access Camera Webcam, 0 = For Internal Webcam, 1 = For Internal Webcam
videoCapture = cv.VideoCapture(0)
#Use Library dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Looping
while True:
    _,frame = videoCapture.read()
    makeGray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    face = detector(makeGray)

    for i in face:
        faceLandmark = predictor(makeGray,i)
        rightEye = []
        leftEye = []

        #Left Eye
        for n in range (36,42):
            x = faceLandmark.part(n).x
            y = faceLandmark.part(n).y
            leftEye.append((x,y))
            #nPoint = next point
            nPoint = n+1
            if n == 41:
                nPoint = 36
            x2 = faceLandmark.part(nPoint).x
            y2 = faceLandmark.part(nPoint).y
            cv.line(frame,(x,y),(x2,y2),(255,0,0),2)
        #Right Eye
        for n in range (42,48):
            x = faceLandmark.part(n).x
            y = faceLandmark.part(n).y
            rightEye.append((x,y))
            #nPoint = next point
            nPoint = n+1
            if n == 47:
                nPoint = 42
            x2 = faceLandmark.part(nPoint).x
            y2 = faceLandmark.part(nPoint).y
            cv.line(frame,(x,y),(x2,y2),(255,0,0),2)

        leftEAR = calculatedEAR(leftEye)
        rightEAR = calculatedEAR(rightEye)

        EAR = (leftEAR+rightEAR)/2
        EAR = round(EAR,2)

        if EAR < earValue:
            cv.putText(frame,topText,(80,60),cv.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
            print("You're Sleppy")
        print(EAR)


    cv.imshow(windowName,frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

videoCapture.release()
cv.destroyAllWindows()