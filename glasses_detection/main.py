import numpy as np
import  cv2

GLASSES_THRESHOLD = 204
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


def main():
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

            eyes_normal = eye_cascade.detectMultiScale(img_eye_gray)#どちらの目でもいいから検出？
            eyes_glasses = eyeglasses_cascade.detectMultiScale(img_eye_gray)#眼鏡検出？のはず...
            eyes_left = eye_l_cascade.detectMultiScale(img_eye_l_gray)#左目検出
            #eyes_left = eye_l_cascade.detectMultiScale(img_eye_gray)
            eyes_right = eye_r_cascade.detectMultiScale(img_eye_r_gray)#右目検出
            #eyes_right = eye_r_cascade.detectMultiScale(img_eye_gray)

            eyes_all = list(eyes_normal) + list(eyes_glasses) + list(eyes_left) + list(eyes_right)

            # 検出した目が2個以上なら
            if len(eyes_all) >= 2:
                # 目の座標・右目距離・左目距離を計算
                eyePoints, rightEyeDistances, leftEyeDistances = getEyePointsAndDistances(eyes_all, (int(w/4), int(h/4)), (int(w*3/4), int(h/4)))

                # 左右の目に最も近い目を決定
                rightEyePos = eyePoints[rightEyeDistances.index(min(rightEyeDistances))]
                leftEyePos = eyePoints[leftEyeDistances.index(min(leftEyeDistances))]

            # 
            if (0 <= rightEyePos[0] < w/2) and (0 <= rightEyePos[1] < h/2) and (w/2 <= leftEyePos[0] < w) and (0 <= leftEyePos[1] < h/2):
                # 同じ目を指していたら
                #if rightEyePos is leftEyePos:
                #    # 既存の最小距離を最大距離に変更
                #    maxDistance = max(max(rightEyeDistances), max(leftEyeDistances))
                #    rightEyeDistances[eyePoints.index(rightEyePos)] += maxDistance
                #    leftEyeDistances[eyePoints.index(leftEyePos)] += maxDistance
                #
                #    # 片目ごとに2番目に距離の短い目に置き換え、その合計距離の短い方を選択
                #    if (rightEyeDistances[eyePoints.index(rightEyePos)] + min(leftEyeDistances)) < (leftEyeDistances[eyePoints.index(leftEyePos)] + min(rightEyeDistances)):
                #        leftEyePos = eyePoints[leftEyeDistances.index(min(leftEyeDistances))]
                #    else:
                #        rightEyePos = eyePoints[rightEyeDistances.index(min(rightEyeDistances))]

                # めがね検出
                if detectGlasses(img_eye_gray, rightEyePos, leftEyePos, img_eye):
                    cv2.putText(img_eye, "GLASSES", (0, h), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img_eye, "NOT GLASSES", (0, h), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            
            for (ex, ey, ew, eh) in eyes_normal:
                cv2.rectangle(img_eye, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
            for (ex, ey, ew, eh) in eyes_glasses:
                cv2.rectangle(img_eye, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)
            for (ex, ey, ew, eh) in eyes_left:
                cv2.rectangle(img_eye, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 1)
            for (ex, ey, ew, eh) in eyes_right:
                cv2.rectangle(img_eye, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 1)



        cv2.imshow('frame',frame)
        if cv2.waitKey(1000) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


def getEyePointsAndDistances(eyes, rightEyePos, LeftEyePos):
    """
    目の座標・右目距離・左目距離を返す。

    eyes : 目の座標のタプル
    rightEyePos : 右目の座標
    LeftEyePos : 左目の座標

    return 目の座標・右目距離・左目距離（それぞれがリスト）
    """
    points = []
    rightEyeDistances = []
    leftEyeDistances = []
    for (x, y, w, h) in eyes:
        point = (int(x + w/2), int(y + h/2))
        points.append(point)
        rightEyeDistances.append(getDistance2(point, rightEyePos))
        leftEyeDistances.append(getDistance2(point, LeftEyePos))
    return points, rightEyeDistances, leftEyeDistances


def getDistance2(p1, p2):
    """
    2点間の距離の2乗を計算する。

    p1, p2 : 点の座標のタプル (x, y)
    """
    return ((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2)


def clip(x, min, max):
    """
    xをmin以上max以下の値にする。
    """
    if x <= min:
        return min
    if x >= max:
        return max
    return x


def detectGlasses(img, eye1Pos, eye2Pos, debugImg = None):
    """
    めがねが存在するか判定する。

    img : 顔画像（グレースケール）
    eyeXPos : X個目の目の座標のタプル (x, y)
    debugImg : デバッグ情報を書く画像（省略可）

    return True / False
    """

    # 目の中心座標の計算
    eyeCenter = ((eye1Pos[0] + eye2Pos[0])/2, (eye1Pos[1] + eye2Pos[1])/2)

    # 目のX方向の距離を計算
    eyeDistance = abs(eye1Pos[0] - eye2Pos[0])
    if eyeDistance < min(img.shape[0], img.shape[1])/20:
        eyeDistance = int(min(img.shape[0], img.shape[1])/20)

    # 左右2分割
    split_x = int(eyeCenter[0])
    img_leftFace = img[0 : img.shape[0], 0 : split_x]
    img_rightFace = img[0 : img.shape[0], split_x : img.shape[1]]

    # 左 半楕円マスク
    x = np.array(range(img_leftFace.shape[0]))
    x = (img_leftFace.shape[1] - img_leftFace.shape[1] * np.sqrt(1 - ((x - int(img_leftFace.shape[0]/2))/(img_leftFace.shape[0]/2))**2)).astype(np.int64)
    c = 0
    for y in range(img_leftFace.shape[0]):
        for i in range(x[y]):
            img_leftFace[y, i] = c
            if c == 255:
                c = 0
            else:
                c += 1

    # 右 半楕円マスク
    x = np.array(range(img_rightFace.shape[0]))
    x = (img_rightFace.shape[1] * np.sqrt(1 - ((x - int(img_rightFace.shape[0]/2))/(img_rightFace.shape[0]/2))**2)).astype(np.int64)
    c = 0
    for y in range(img_rightFace.shape[0]):
        for i in range(x[y], img_rightFace.shape[1]):
            img_rightFace[y, i] = c
            if c == 255:
                c = 0
            else:
                c += 1

    # 画像の2値化
    ret, img_leftFace_2 = cv2.threshold(img_leftFace, 0, 255, cv2.THRESH_OTSU)
    ret, img_rightFace_2 = cv2.threshold(img_rightFace, 0, 255, cv2.THRESH_OTSU)

    #　分割した画像を連結
    img_2 = cv2.hconcat([img_leftFace_2, img_rightFace_2])

    # 目の間周辺の画像を切り出し
    x2 = clip(int(eyeCenter[0] + eyeDistance/4), 0, img.shape[1])
    x1 = clip(int(eyeCenter[0] - eyeDistance/4), 0, img.shape[1])
    y1 = clip(int(eyeCenter[1] - eyeDistance/4), 0, img.shape[0])
    y2 = clip(int(eyeCenter[1] + eyeDistance/4), 0, img.shape[0])
    img_betweenEyes = img_2[y1 : y2, x1 : x2]

    # 平均の明るさを計算
    average = np.mean(img_betweenEyes)

    # デバッグ用
    if debugImg is not None:
        debugImg[:, :, 0] = img_2
        debugImg[:, :, 1] = img_2
        debugImg[:, :, 2] = img_2
        cv2.line(debugImg, eye1Pos, eye2Pos, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(debugImg, (x1, y1), (x2, y2), (255, 0, 0), 1)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        img_hist = np.zeros((256, 256, 3), dtype = np.uint8)
        for i in range(256):
            for j in range(int(256 - 256 * hist[i] / max(hist))):
                img_hist[j][i][0] = 255
                img_hist[j][i][1] = 255
                img_hist[j][i][2] = 255
        cv2.imshow("Histogram", img_hist)

    if average <= GLASSES_THRESHOLD:
        return True
    else:
        return False


if __name__ == '__main__':
    main()
