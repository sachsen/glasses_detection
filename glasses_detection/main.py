import numpy as np
import  cv2

GLASSES_THRESHOLD = 0.5
HAAR_FILE="haarcascade_frontalface_default.xml"
HAAR_FILE2="haarcascade_eye.xml"
cascade=cv2.CascadeClassifier(HAAR_FILE)
eye_cascade=cv2.CascadeClassifier(HAAR_FILE2)
capture = cv2.VideoCapture(0)


while(True):
    ret, frame = capture.read()
    img_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cascade.detectMultiScale(img_g)
    for (x, y, w, h) in face:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        img_eye_gray = img_g[y:y + h, x:x + w]
        img_eye = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(img_eye_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img_eye, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)


    cv2.imshow('frame',frame)
    if cv2.waitKey(1000) == 27:
        break

capture.release()
cv2.destroyAllWindows()


def detectGlasses(img, eye1Pos, eye2Pos):
    """
    めがねが存在するか判定する。

    img : 顔画像（グレースケール）
    eyeXPos : X個目の目の座標のタプル (x, y)

    return True / False
    """

    # 目の中心座標の計算
    eyeCenter = ((eye1Pos[0] + eye2Pos[0])/2, (eye1Pos[1] + eye2Pos[1])/2)

    # 目のX方向の距離を計算
    eyeDistance = abs(eye1Pos[0] - eye2Pos[0])

    # 画像の2値化
    ret, img_2 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    # 目の間周辺の画像を切り出し
    img_betweenEyes = img_2[eyeCenter[1] - eyeDistance/4 : eyeCenter[1] + eyeDistance/4, eyeCenter[0] - eyeDistance/4 : eyeCenter[0] + eyeDistance/4]

    # 平均の明るさを計算
    average = np.mean(img_betweenEyes)

    if average >= GLASSES_THRESHOLD:
        return True
    else:
        return False
