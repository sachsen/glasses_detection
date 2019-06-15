import numpy as np
import  cv2

GLASSES_THRESHOLD = 0.5
HAAR_FILE="haarcascade_frontalface_default.xml"
HAAR_FILE2="haarcascade_eye.xml"
cascade=cv2.CascadeClassifier(HAAR_FILE)
eye_cascade=cv2.CascadeClassifier(HAAR_FILE2)
capture = cv2.VideoCapture(0)


def main():
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
            
            # 検出した目が2個以上なら
            # TODO: 3個以上の目を検出したときに、その中から2個を選ぶ処理を追加する
            if len(eyes) >= 2:
                if detectGlasses(img_eye_gray, (eyes[0][0] + eyes[0][2]/2, eyes[0][1] + eyes[0][3]/2), (eyes[1][0] + eyes[1][2]/2, eyes[1][1] + eyes[1][3]/2)):
                    cv2.putText(img_eye, "GLASSES", (0, h), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img_eye, "NOT GLASSES", (0, h), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)



        cv2.imshow('frame',frame)
        if cv2.waitKey(1000) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


def clip(x, min, max):
    """
    xをmin以上max以下の値にする。
    max < min なら min が返る。
    """
    if x <= min:
        return min
    if x >= max:
        return max
    return x


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
    x2 = clip(int(eyeCenter[0] + eyeDistance/4), 0, img.shape[1])
    x1 = clip(int(eyeCenter[0] - eyeDistance/4), 0, img.shape[1])
    y1 = clip(int(eyeCenter[1] - eyeDistance/4), 0, img.shape[0])
    y2 = clip(int(eyeCenter[1] + eyeDistance/4), 0, img.shape[0])
    img_betweenEyes = img_2[y1 : y2, x1 : x2]

    # 平均の明るさを計算
    average = np.mean(img_betweenEyes)

    if average >= GLASSES_THRESHOLD:
        return True
    else:
        return False


main()
