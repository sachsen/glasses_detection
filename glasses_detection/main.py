import  cv2
HAAR_FILE="haarcascade_frontalface_default.xml"
HAAR_FILE="haarcascade_eye.xml"
cascade=cv2.CascadeClassifier(HAAR_FILE)

capture = cv2.VideoCapture(0)
while(True):
    ret, frame = capture.read()
    img_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cascade.detectMultiScale(img_g)
    for (x, y, w, h) in face:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.imshow('frame',frame)
    if cv2.waitKey(10) == 27:
        break

capture.release()
cv2.destroyAllWindows()