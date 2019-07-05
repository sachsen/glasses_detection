import numpy as np
import  cv2
import os
from main import getBothEyes, getGlassesValue

THRESHOLD_SPLIT = 20

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

                # 両目を取得
                goodEyes, eye1Pos, eye2Pos = getBothEyes((w, h), eyes_all)

                if goodEyes:
                    # めがね
                    value = int(getGlassesValue(img_eye, img_eye_gray, eye1Pos, eye2Pos))
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
            for k in range(int(200 * thrshes[j] / len(files[i]))):
                print("■", end = "")
            print()
        print()


if __name__ == '__main__':
    main()
