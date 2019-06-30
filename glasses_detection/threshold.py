import numpy as np
import  cv2
import os

THRESHOLD_SPLIT = 10

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
eye_cascade=cv2.CascadeClassifier(HAAR_FILE2)


def main():
    files = [[], []]

    # めがねとめがねでない画像
    d = ("./Image_Glasses", "./Image_NoGlasses")
    for i in range(2):
        if os.path.exists(d[i]):
            for f in os.listdir(d[i]):
                n, e = os.path.splitext(f)
                if "jpg" in e:
                    files[i].append(os.path.join(d[i], f))

    
    for i in range(2):
        if i == 0:
            print("めがね画像")
        elif i == 1:
            print("めがねでない画像")

        print("枚数：", len(files[i]))

        thrshes = np.zeros(THRESHOLD_SPLIT + 1, int).tolist()
        faceNum = 0

        for file in files[i]:
            frame = cv2.imread(file)
            img_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = cascade.detectMultiScale(img_g)
            for (x, y, w, h) in face:
                faceNum += 1

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

                    # 同じ目を指していたら
                    if rightEyePos is leftEyePos:
                        # 既存の最小距離を最大距離に変更
                        maxDistance = max(max(rightEyeDistances), max(leftEyeDistances))
                        rightEyeDistances[eyePoints.index(rightEyePos)] += maxDistance
                        leftEyeDistances[eyePoints.index(leftEyePos)] += maxDistance

                        # 片目ごとに2番目に距離の短い目に置き換え、その合計距離の短い方を選択
                        if (rightEyeDistances[eyePoints.index(rightEyePos)] + min(leftEyeDistances)) < (leftEyeDistances[eyePoints.index(leftEyePos)] + min(rightEyeDistances)):
                            leftEyePos = eyePoints[leftEyeDistances.index(min(leftEyeDistances))]
                        else:
                            rightEyePos = eyePoints[rightEyeDistances.index(min(rightEyeDistances))]

                    # めがね
                    value = int(getGlassesValue(img_eye_gray, rightEyePos, leftEyePos))
                    thrshes[int(THRESHOLD_SPLIT * value / 255)] += 1
                
                for (ex, ey, ew, eh) in eyes_normal:
                    cv2.rectangle(img_eye, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
                for (ex, ey, ew, eh) in eyes_glasses:
                    cv2.rectangle(img_eye, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)
                for (ex, ey, ew, eh) in eyes_left:
                    cv2.rectangle(img_eye, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 1)
                for (ex, ey, ew, eh) in eyes_right:
                    cv2.rectangle(img_eye, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 1)
                



            
        print("検出した顔：", faceNum)
        print("めがね値：")
        for j in range(len(thrshes)):
            print(str(j).zfill(2), " ", end = "")
            for k in range(int(10 * thrshes[j] / len(files[i]))):
                print("■", end = "")
            print()
        print()


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


def getGlassesValue(img, eye1Pos, eye2Pos, debugImg = None):
    """
    めがね値を返す。

    img : 顔画像（グレースケール）
    eyeXPos : X個目の目の座標のタプル (x, y)
    debugImg : デバッグ情報を書く画像（省略可）

    return めがね値
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

    return average


if __name__ == '__main__':
    main()
