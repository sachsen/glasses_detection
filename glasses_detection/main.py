import numpy as np
import cv2
import copy
GLASSES_THRESHOLD = 3
BLUE_CUT_GLASSES_THRESHOLD = 10
BLUE_CUT_GLASSES_THRESHOLD2 = 5
DIFF=False#背景との差分でマスウを生成するかどうか。するなら最初に背景を取るために動画を流し、被写体がいないタイミングでescを押す。
#正面を向いた時、ブルーライトの検出率が悪くなるので、２つ目を検出した時の閾値を別に用意している。
HAAR_FILE = "haarcascade_frontalface_default.xml"
HAAR_FILE2 = "haarcascade_eye_tree_eyeglasses.xml"
HAAR_FILE3= "haarcascade_eye.xml"
HAAR_FILE4= "haarcascade_lefteye_2splits.xml"
HAAR_FILE5= "haarcascade_righteye_2splits.xml"
cascade = cv2.CascadeClassifier(HAAR_FILE)
eye_cascade = cv2.CascadeClassifier(HAAR_FILE2)
eye_cascade2 = cv2.CascadeClassifier(HAAR_FILE3)
eye_cascade3 = cv2.CascadeClassifier(HAAR_FILE4)
eye_cascade4 = cv2.CascadeClassifier(HAAR_FILE5)
capture = cv2.VideoCapture(0)

def main():

    ret, backframe = capture.read()  # スコプを伸ばすためのの記述
    while (DIFF):#背景画像を取得するので、被写体や動く物体はカメラから見えない所に置いてEscキー押してください。
        ret, backframe = capture.read()
        cv2.imshow("satuei",backframe)
        if cv2.waitKey(10) == 27:
            break
    #capture.release()
    cv2.destroyAllWindows()

    while (True):
        ret, frame = capture.read()
        processed,mask=prepareDetection(backframe,frame)
        frame=processed
        frame_m=copy.deepcopy(frame)
        frame_m[mask == 0] = [0, 0, 0]
        img_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = cascade.detectMultiScale(img_g)

        # めがね判定オーバーレイ表示用画像
        frame_over = np.zeros(frame.shape, dtype = np.uint8)

        for (x, y, w, h) in face:
            if DIFF:#置き換え
                img_g = cv2.cvtColor(frame_m, cv2.COLOR_BGR2GRAY)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            img_eye_gray = img_g[y:y + h, x:x + w]
            img_eye = frame[y:y + h, x:x + w]
            img_upper_face=frame[y+int(h/4):y + int(h/2), x+int(w/11*2):x + int(w/11*9)]
            blue= detectBluelightCutGlasses(img_upper_face,frame,10,10)
            #print(blue)

            #以下cascade祭り
            eyes = eye_cascade.detectMultiScale(img_eye_gray)
            eyes2 = eye_cascade2.detectMultiScale(img_eye_gray)
            eyes3 = eye_cascade3.detectMultiScale(img_eye_gray)
            eyes4 = eye_cascade4.detectMultiScale(img_eye_gray)
            #それぞれの検出した目をeyesに集積
            if(len(eyes)>0 and len(eyes2)>0):
                eyes = np.vstack((eyes, eyes2))
            if (len(eyes) > 0 and len(eyes3) > 0):
                eyes = np.vstack((eyes, eyes3))
            if (len(eyes) > 0 and len(eyes4) > 0):
                eyes = np.vstack((eyes, eyes4))
            # 検出した目が2個以上なら
            if (len(eyes))>= 2:
                # 目の座標・右目距離・左目距離を計算
                eyePoints, rightEyeDistances, leftEyeDistances = getEyePointsAndDistances(eyes, (int(w/4), int(h/4)), (int(w*3/4), int(h/4)))
                # 左右の目に最も近い目を決定
                rightEyePos = eyePoints[rightEyeDistances.index(min(rightEyeDistances))]
                leftEyePos = eyePoints[leftEyeDistances.index(min(leftEyeDistances))]

                #
                if (0 <= rightEyePos[0] < w/2) and (0 <= rightEyePos[1] < h/2) and (w/2 <= leftEyePos[0] < w) and (0 <= leftEyePos[1] < h/2):
                    # めがね検出
                    if not detectGlasses(eyes,img_eye_gray,img_eye, rightEyePos, leftEyePos, img_eye):
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
            
            if blue>BLUE_CUT_GLASSES_THRESHOLD:#ブルーライトカット眼鏡検出1

                cv2.putText(img_upper_face, "Bluelight Cut Glasses", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img_eye, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
            # めがね判定オーバーレイ表示用画像重ね合わせ
            frame = np.clip((frame + 0.9 * frame_over), 0, 255).astype(np.uint8)

        cv2.imshow('frame',frame)
        if cv2.waitKey(10) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()

def prepareDetection(bgimg,img):
    if(img is not None):
        framemask=backgroundMask(bgimg,img)
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_3 = cv2.adaptiveThreshold(img_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 155, 30)
    img_3_3 = cv2.cvtColor(img_3, cv2.COLOR_GRAY2BGR)
    img_3_3 = cv2.GaussianBlur(img_3_3, (11, 11), 12)
    img_diff = cv2.absdiff(img, img_3_3)  # 差分計算

    # img_diff[np.where((img_diff == [255,255,255]).all(axis=2))] = [240/2,221/2,195/2]#色の塗りつぶし(色の変換)

    img = cv2.addWeighted(img, 0.95, img_diff, 0.05, 3)  # 画像合成
    # img=cv2.add(img,img_diff)

    return  img,framemask
def detectBluelightCutGlasses(img,img_face,s,v):#svはhsvのsv
    # 青い眼鏡、青い髪、青い入れ墨等は誤認識します。
    #青色の範囲をHSVで指定して収集(frame_mask)
    #全体に青みがかってる場合の調整値を求める
    img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2HSV)
    lower = np.array([75, s, v])
    upper = np.array([135, 100, 100])
    frame_mask2 = cv2.inRange(img_face, lower, upper)
    average2 = np.mean(frame_mask2)
    #調整ありきで目の周辺の青色取得
    img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower = np.array([75, s+int(average2), v+int(average2)])
    upper = np.array([135, 100, 100])
    frame_mask = cv2.inRange(img, lower, upper)
    average = np.mean(frame_mask)


    cv2.imshow("bluelight",frame_mask)#確認用
    return average

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
        point = (int(x + w / 2), int(y + h / 2))
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


def detectGlasses(eyes,img,img_color, eye1Pos, eye2Pos, debugImg = None):
    """
    めがねが存在するか判定する。

    img : 顔画像（グレースケール）
    eyeXPos : X個目の目の座標のタプル (x, y)
    debugImg : デバッグ情報を書く画像（省略可）

    return True / False
    """
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    lower = np.array([-33/2, 0, 50])
    upper = np.array([64/2, 100, 100])
    frame_mask = cv2.inRange(img_hsv, lower, upper)
    frame_mask = cv2.GaussianBlur(frame_mask, (21, 21), 12)
    frame_mask=cv2.cvtColor(frame_mask,cv2.COLOR_GRAY2BGR)
    img_color=cv2.addWeighted(img_color,0.9,frame_mask,0.1,3)
    # 目の中心座標の計算
    eyeCenter = ((eye1Pos[0] + eye2Pos[0]) / 2, (eye1Pos[1] + eye2Pos[1]) / 2)

    # 目のX方向の距離を計算
    eyeDistance = abs(eye1Pos[0] - eye2Pos[0])
    if eyeDistance < min(img.shape[0], img.shape[1]) / 20:
        eyeDistance = int(min(img.shape[0], img.shape[1]) / 20)

    # 画像のエッジを求める
    img_2 = cv2.Canny(img, 50, 250)

    for (ex, ey, ew, eh) in eyes:#目の辺りを黒塗りにして差をつける
        img_2 = cv2.rectangle(img_2, (ex, ey), (ex + ew, ey + eh), (0, 0, 0), cv2.FILLED)

    # 目の周辺を切り出し2
    x4 = clip(int(eyeCenter[0] + eyeDistance/7), 0, img.shape[1])
    x3 = clip(int(eyeCenter[0] - eyeDistance/7), 0, img.shape[1])
    y3 = clip(int(eyeCenter[1] - eyeDistance ), 0, img.shape[0])
    y4 = clip(int(eyeCenter[1] + eyeDistance ), 0, img.shape[0])
    #img_2 = cv2.rectangle(img_2, (int(img_2.shape[0]/10*4), 0), (int(img_2.shape[0]/10*6), img_2.shape[1]), (0, 0, 0), cv2.FILLED)  # 鼻の線を消したい。
    img_2 = cv2.rectangle(img_2, (x3, y3), (x4, y4), (0, 0, 0), cv2.FILLED)

    cv2.imshow("img_2", img_2)
    # 目の間周辺の画像を切り出し
    x2 = clip(int(eyeCenter[0] + eyeDistance), 0, img.shape[1])
    x1 = clip(int(eyeCenter[0] - eyeDistance), 0, img.shape[1])
    y1 = clip(int(eyeCenter[1] - eyeDistance / 2), 0, img.shape[0])
    y2 = clip(int(eyeCenter[1] + eyeDistance / 2), 0, img.shape[0])
    img_betweenEyes = img_2[y1:y2, x1:x2]
    img_betweenEyes_color = img_color[y1:y2, x1:x2]

    #ブルーライトカット眼鏡検出2
    blue = detectBluelightCutGlasses(img_betweenEyes_color,img_color, 10, 15)
    print(str(blue) + "blue")
    if blue > BLUE_CUT_GLASSES_THRESHOLD2:
        cv2.putText(img_color, "Bluelight Cut Glasses", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
    # 平均の明るさを計算
    average = np.mean(img_betweenEyes)
    print(average)

    # デバッグ用
    if debugImg is not None:
        img_2=cv2.cvtColor(img_2,cv2.COLOR_GRAY2BGR)
        debugImg=cv2.addWeighted(img_2,0.5,img_color,0.5,3)
        '''debugImg[:, :, 0] = img_2
        debugImg[:, :, 1] = img_2
        debugImg[:, :, 2] = img_2'''

        cv2.line(debugImg, eye1Pos, eye2Pos, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(debugImg, (x1, y1), (x2, y2), (255, 0, 0), 1)

    if average >= GLASSES_THRESHOLD:
        return True
    else:
        return False

def backgroundMask(bgimg,img):#マスクを作る
    img_diff = cv2.absdiff(bgimg, img)#背景との差分
    img_diff = cv2.GaussianBlur(img_diff, (51, 51), 12)#ノイズ除去
    img_diff=cv2.cvtColor(img_diff,cv2.COLOR_BGR2GRAY)
    et, img_diff = cv2.threshold(img_diff, 20, 255, cv2.THRESH_BINARY)#二値化
    kernel = np.ones((3, 3), dtype=np.uint8)
    for i in range(10):#黒い穴を埋める
        img_diff = cv2.dilate(img_diff, kernel)  # 白が膨張
        img_diff = cv2.erode(img_diff, kernel)  # 黒が膨張

    img_diff = cv2.morphologyEx(img_diff, cv2.MORPH_CLOSE, kernel)  # モルフォロジー。第二で方式を選択。
    if DIFF:
        cv2.imshow("img_diff",img_diff)
    return img_diff

if __name__ == '__main__':
    main()