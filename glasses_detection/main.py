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
        faceSize = cascade.detectMultiScale(img_g)

        # めがね判定オーバーレイ表示用画像
        frame_over = np.zeros(frame.shape, dtype = np.uint8)

        for (x, y, w, h) in faceSize:
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

            # 両目を取得
            goodEyes, eye1Pos, eye2Pos = getBothEyes((w, h), eyes_all)

            if goodEyes:
                # めがね検出
                if getGlassesValue(img_eye, img_eye_gray, eye1Pos, eye2Pos, 1) <= GLASSES_THRESHOLD:
                    #cv2.putText(img_eye, "GLASSES", (0, h), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame_over, (int(x + w/2), int(y + h/2)), int(0.35*(w+h)), (0, 255, 0), thickness = int(0.05*(w+h)), lineType = cv2.LINE_AA)
                else:
                    #cv2.putText(img_eye, "NOT GLASSES", (0, h), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                    s = int(0.03 * (w + h))
                    t = s * 2
                    wh = int(w/2)
                    hh = int(h/2)
                    p = np.array([[x + s, y - s], [x + wh, y + hh - t], [x + w - s, y - s], [x + w + s, y + s], [x + wh + t, y + hh], [x + w + s, y + h - s], [x + w - s, y + h + s], [x + wh, y + hh + t], [x + s, y + h + s], [x - s, y + h - s], [x + wh - t, y + hh], [x - s, y + s]]).reshape(1, -1, 2)
                    cv2.fillPoly(frame_over, p, (0, 0, 255))
            
            for (ex, ey, ew, eh) in eyes_normal:
                cv2.rectangle(img_eye, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
            for (ex, ey, ew, eh) in eyes_glasses:
                cv2.rectangle(img_eye, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)
            for (ex, ey, ew, eh) in eyes_left:
                cv2.rectangle(img_eye, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 1)
            for (ex, ey, ew, eh) in eyes_right:
                cv2.rectangle(img_eye, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 1)

        # めがね判定オーバーレイ表示用画像重ね合わせ
        frame = np.clip((frame + 0.9 * frame_over), 0, 255).astype(np.uint8)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1000) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


def clip(x, min, max):
    """
    xをmin以上max以下の値にする。
    """
    if x <= min:
        return min
    if x >= max:
        return max
    return x


def getBothEyes(faceSize, eyes):
    """
    顔座標・目座標から両目を取得する。
    顔座標は (w, h)
    目座標は顔画像の左上を基準にした (x, y, w, h)

    return , eye1Pos, eye2Pos
    """

    if len(eyes) == 0:
        return False, (0, 0), (0, 0)

    # 目の座標 (x, y) を計算
    eyes = np.array(eyes)
    eyesPos = (eyes[:, 0 : 2] + eyes[:, 2 : 4] / 2).astype(np.int32)

    # 両目が存在するであろう点との距離を計算
    eye1Distances = np.sqrt((eyesPos[:, 0] - int(    faceSize[0] / 4)) ** 2 + (eyesPos[:, 1] - int(faceSize[1] / 4))**2)
    eye2Distances = np.sqrt((eyesPos[:, 0] - int(3 * faceSize[0] / 4)) ** 2 + (eyesPos[:, 1] - int(faceSize[1] / 4))**2)

    # 両目を選ぶ
    eye1Pos = tuple(eyesPos[np.argmin(eye1Distances)].tolist())
    eye2Pos = tuple(eyesPos[np.argmin(eye2Distances)].tolist())

    # 右上1/4の範囲に右目が、左上1/4の範囲に左目があるか
    goodEyes = (0 <= eye1Pos[0] < faceSize[0] / 2) and (0 <= eye1Pos[1] < faceSize[1] / 2) and (faceSize[0] / 2 <= eye2Pos[0] < faceSize[0]) and (0 <= eye2Pos[1] < faceSize[1] / 2)

    return goodEyes, eye1Pos, eye2Pos


def getGlassesValue(img_c, img_g, eye1Pos, eye2Pos, debugCode = 0):
    """
    めがね値を返す。

    img_c : 顔画像（カラー）
    img_g : 顔画像（グレースケール）
    eyeXPos : X個目の目の座標のタプル (x, y)
    debugCode : 0 -> 何もしない, 1 -> デバッグ情報表示

    return めがね値
    """

    # 目の中心座標の計算
    eyeCenter = ((eye1Pos[0] + eye2Pos[0])/2, (eye1Pos[1] + eye2Pos[1])/2)

    # 目のX方向の距離を計算
    eyeDistance = abs(eye1Pos[0] - eye2Pos[0])
    if eyeDistance < min(img_g.shape[0], img_g.shape[1])/20:
        eyeDistance = int(min(img_g.shape[0], img_g.shape[1])/20)

    # 左右2分割
    split_x = int(eyeCenter[0])
    img_leftFace = img_g[0 : img_g.shape[0], 0 : split_x]
    img_rightFace = img_g[0 : img_g.shape[0], split_x : img_g.shape[1]]

    # 左 半楕円マスク
    x = np.array(range(img_leftFace.shape[0]))
    x = (img_leftFace.shape[1] - img_leftFace.shape[1] * np.sqrt(1 - ((x - int(img_leftFace.shape[0]/2))/(img_leftFace.shape[0]/2))**2)).astype(np.int32)
    for y in range(img_leftFace.shape[0]):
        img_leftFace[y, 0 : x[y]] = 0

    # 右 半楕円マスク
    x = np.array(range(img_rightFace.shape[0]))
    x = (img_rightFace.shape[1] * np.sqrt(1 - ((x - int(img_rightFace.shape[0]/2))/(img_rightFace.shape[0]/2))**2)).astype(np.int32)
    for y in range(img_rightFace.shape[0]):
        img_rightFace[y, x[y] : img_rightFace.shape[1]] = 0

    # 画像の2値化
    ret, img_leftFace_2 = cv2.threshold(img_leftFace, 0, 255, cv2.THRESH_OTSU)
    ret, img_rightFace_2 = cv2.threshold(img_rightFace, 0, 255, cv2.THRESH_OTSU)

    #　分割した画像を連結
    img_2 = cv2.hconcat([img_leftFace_2, img_rightFace_2])

    # 目の間周辺の画像を切り出し
    x2 = clip(int(eyeCenter[0] + eyeDistance/4), 0, img_g.shape[1])
    x1 = clip(int(eyeCenter[0] - eyeDistance/4), 0, img_g.shape[1])
    y1 = clip(int(eyeCenter[1] - eyeDistance/4), 0, img_g.shape[0])
    y2 = clip(int(eyeCenter[1] + eyeDistance/4), 0, img_g.shape[0])
    img_betweenEyes = img_2[y1 : y2, x1 : x2]

    # 平均の明るさを計算
    average = np.mean(img_betweenEyes)

    # デバッグ用
    if debugCode != 0:
        img_c[:, :, 0] = img_2
        img_c[:, :, 1] = img_2
        img_c[:, :, 2] = img_2
        cv2.line(img_c, eye1Pos, eye2Pos, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(img_c, (x1, y1), (x2, y2), (255, 0, 0), 1)
        hist = cv2.calcHist([img_g], [0], None, [256], [0, 256])
        img_hist = np.zeros((256, 256, 3), dtype = np.uint8)
        for i in range(256):
            for j in range(int(256 - 256 * hist[i] / max(hist))):
                img_hist[j][i][0] = 255
                img_hist[j][i][1] = 255
                img_hist[j][i][2] = 255
        cv2.imshow("Histogram", img_hist)

    return average


if __name__ == '__main__':
    main()
