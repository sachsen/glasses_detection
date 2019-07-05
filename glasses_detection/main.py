import numpy as np
import cv2

GLASSES_THRESHOLD = 11
HAAR_FILE = "haarcascade_frontalface_default.xml"
HAAR_FILE2 = "haarcascade_eye.xml"
cascade = cv2.CascadeClassifier(HAAR_FILE)
eye_cascade = cv2.CascadeClassifier(HAAR_FILE2)
capture = cv2.VideoCapture(0)


def main():
    while (True):
        ret, frame = capture.read()
        img_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = cascade.detectMultiScale(img_g)
        for (x, y, w, h) in face:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255),
                                  1)
            img_eye_gray = img_g[y:y + h, x:x + w]
            img_eye = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(img_eye_gray)

            # 検出した目が2個以上なら
            if len(eyes) >= 2:
                # 目の座標・右目距離・左目距離を計算
                eyePoints, rightEyeDistances, leftEyeDistances = getEyePointsAndDistances(
                    eyes, (int(w / 4), int(h / 4)),
                    (int(w * 3 / 4), int(h / 4)))

                # 左右の目に最も近い目を決定
                rightEyePos = eyePoints[rightEyeDistances.index(
                    min(rightEyeDistances))]
                leftEyePos = eyePoints[leftEyeDistances.index(
                    min(leftEyeDistances))]

                # 同じ目を指していたら
                if rightEyePos is leftEyePos:
                    # 既存の最小距離を最大距離に変更
                    maxDistance = max(max(rightEyeDistances),
                                      max(leftEyeDistances))
                    rightEyeDistances[eyePoints.index(
                        rightEyePos)] += maxDistance
                    leftEyeDistances[eyePoints.index(
                        leftEyePos)] += maxDistance

                    # 片目ごとに2番目に距離の短い目に置き換え、その合計距離の短い方を選択
                    if (rightEyeDistances[eyePoints.index(rightEyePos)] +
                            min(leftEyeDistances)) < (
                                leftEyeDistances[eyePoints.index(leftEyePos)] +
                                min(rightEyeDistances)):
                        leftEyePos = eyePoints[leftEyeDistances.index(
                            min(leftEyeDistances))]
                    else:
                        rightEyePos = eyePoints[rightEyeDistances.index(
                            min(rightEyeDistances))]

                # めがね検出
                if detectGlasses(img_eye_gray, rightEyePos, leftEyePos,
                                 img_eye):
                    cv2.putText(img_eye, "GLASSES", (0, h),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),
                                2, cv2.LINE_AA)
                else:
                    cv2.putText(img_eye, "NOT GLASSES", (0, h),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),
                                2, cv2.LINE_AA)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img_eye, (ex, ey), (ex + ew, ey + eh),
                              (0, 255, 0), 1)

        cv2.imshow('frame', frame)
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
    return ((p1[0] - p2[0])**2) + ((p1[1] - p2[1])**2)


def clip(x, min, max):
    """
		xをmin以上max以下の値にする。
		"""
    if x <= min:
        return min
    if x >= max:
        return max
    return x


def detectGlasses(img, eye1Pos, eye2Pos, debugImg=None):
    """
		めがねが存在するか判定する。

		img : 顔画像（グレースケール）
		eyeXPos : X個目の目の座標のタプル (x, y)
		debugImg : デバッグ情報を書く画像（省略可）

		return True / False
		"""

    # 目の中心座標の計算
    eyeCenter = ((eye1Pos[0] + eye2Pos[0]) / 2, (eye1Pos[1] + eye2Pos[1]) / 2)

    # 目のX方向の距離を計算
    eyeDistance = abs(eye1Pos[0] - eye2Pos[0])
    if eyeDistance < min(img.shape[0], img.shape[1]) / 20:
        eyeDistance = int(min(img.shape[0], img.shape[1]) / 20)

    # 画像の明るさを正規化
    img = ((img - np.mean(img)) / np.std(img) * 64 + 128).astype(np.uint8)
    # 画像のエッジを求める
    img_2 = cv2.Canny(img, 0, 255)

    # 目の間周辺の画像を切り出し
    x2 = clip(int(eyeCenter[0] + eyeDistance), 0, img.shape[1])
    x1 = clip(int(eyeCenter[0] - eyeDistance), 0, img.shape[1])
    y1 = clip(int(eyeCenter[1] - eyeDistance / 2), 0, img.shape[0])
    y2 = clip(int(eyeCenter[1] + eyeDistance / 2), 0, img.shape[0])
    img_betweenEyes = img_2[y1:y2, x1:x2]

    # 平均の明るさを計算
    average = np.mean(img_betweenEyes)
    print(average)

    # デバッグ用
    if debugImg is not None:
        debugImg[:, :, 0] = img_2
        debugImg[:, :, 1] = img_2
        debugImg[:, :, 2] = img_2
        cv2.line(debugImg, eye1Pos, eye2Pos, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(debugImg, (x1, y1), (x2, y2), (255, 0, 0), 1)

    if average >= GLASSES_THRESHOLD:
        return True
    else:
        return False


if __name__ == '__main__':
    main()
