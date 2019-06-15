import  cv2
HAAR_FILE="haarcascade_frontalface_default.xml"
HAAR_FILE2="haarcascade_eye.xml"
cascade=cv2.CascadeClassifier(HAAR_FILE)
eye_cascade=cv2.CascadeClassifier(HAAR_FILE2)
capture = cv2.VideoCapture(0)
while(True):
    ret, frame = capture.read()
    img_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = eye_cascade.detectMultiScale(img_g)
    for (x, y, w, h) in face:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        img_eye_gray = img_g[y:y + h, x:x + w]
        frame = frame[y:y + h, x:x + w]
        eyes = cascade.detectMultiScale(img_eye_gray)
        for (ex, ey, ew, eh) in eyes:
            frame=cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)


    cv2.imshow('frame',frame)
    if cv2.waitKey(10) == 27:
        break

capture.release()
cv2.destroyAllWindows()