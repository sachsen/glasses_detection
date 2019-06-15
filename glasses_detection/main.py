import  cv2
HAAR_FILE="haarcascade_frontalface_default.xml"
HAAR_FILE2="haarcascade_eye.xml"
HAAR_FILE3="haarcascade_eye_tree_eyeglasses.xml"
HAAR_FILE4="haarcascade_lefteye_2splits.xml"
HAAR_FILE5="haarcascade_righteye_2splits.xml"

cascade=cv2.CascadeClassifier(HAAR_FILE)
eye_cascade=cv2.CascadeClassifier(HAAR_FILE2)
eyeglasses_cascade=cv2.CascadeClassifier(HAAR_FILE3)
eye_l_cascade=cv2.CascadeClassifier(HAAR_FILE4)
eye_r_cascade=cv2.CascadeClassifier(HAAR_FILE5)
capture = cv2.VideoCapture(0)
while(True):
    ret, frame = capture.read()
    img_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cascade.detectMultiScale(img_g)
    for (x, y, w, h) in face:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        img_eye_l_gray = img_g[y:y + int(h/2), x:x + int(w*2/4)]
        img_eye_r_gray =img_g[y:y + int(h/2), x+int(w*2/4):x + w]
        img_eye_gray =img_g[y:y + h, x:x + w]
        img_eye = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(img_eye_gray)#どちらの目でもいいから検出？
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img_eye, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
        glasses = eyeglasses_cascade.detectMultiScale(img_eye_gray)#眼鏡検出？のはず...
        for (ex, ey, ew, eh) in glasses:
            cv2.rectangle(img_eye, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)
        eyes = eye_l_cascade.detectMultiScale(img_eye_l_gray)
        #eyes = eye_l_cascade.detectMultiScale(img_eye_gray)
        for (ex, ey, ew, eh) in eyes:#左目検出
            cv2.rectangle(img_eye, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 1)
        eyes = eye_r_cascade.detectMultiScale(img_eye_r_gray)
        #eyes = eye_r_cascade.detectMultiScale(img_eye_gray)
        for (ex, ey, ew, eh) in eyes:#右目検出
            cv2.rectangle(img_eye, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 1)

    cv2.imshow('frame',frame)
    if cv2.waitKey(10) == 27:
        break

capture.release()
cv2.destroyAllWindows()