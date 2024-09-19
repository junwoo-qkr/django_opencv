from django.conf import settings
import numpy as np # type: ignore
import cv2 # type: ignore

def cv_detect_face(path):
    img = cv2.imread(path, 1)
    if (type(img) is np.ndarray):
        print(img.shape) # 세로, 가로, 채널
        resize_needed = False
        if img.shape[1] > 640: # ex) 가로(img.shape[1])가 1280일 경우,
            resize_needed = True
            new_w = img.shape[1] * (640.0 / img.shape[1]) # 1280 * (640/1280) = 1280 * 0.5
            new_h = img.shape[0] * (640.0 / img.shape[1]) # 기존 세로 * (640/1280) = 기존 세로 * 0.5
        elif img.shape[0] > 480: # ex) 세로(img.shape[0])가 960일 경우,
            resize_needed = True
            new_w = img.shape[1] * (480.0 / img.shape[0]) # 기존 가로 * (480/960) = 기존 가로 * 0.5
            new_h = img.shape[0] * (480.0 / img.shape[0]) # 960 * (480/960) = 960 * 0.5

        if resize_needed == True:
            img = cv2.resize(img, (int(new_w), int(new_h)))
        # Haar-based Cascade Classifier : AdaBoost 기반 머신러닝 물체 인식 모델
        # 이미지에서 눈, 얼굴 등의 부위를 찾는데 주로 이용
        # 이미 학습된 모델을 OpenCV 에서 제공 (http://j.mp/2qIxrxX)
        baseUrl = settings.MEDIA_ROOT_URL + settings.MEDIA_URL
        face_cascade = cv2.CascadeClassifier(baseUrl+'haarcascade_frontalface_default.xml') # face detector
        eye_cascade = cv2.CascadeClassifier(baseUrl+'haarcascade_eye.xml') # eye detector

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 흑백으로 바뀐 이미지, rgb가 bgr로 바뀜
        # detectMultiScale(Original img, ScaleFactor, minNeighbor) : further info. @ http://j.mp/2SxjtKR
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) # 얼굴로 인식된 좌표의 list
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2) # 원본 이미지 위의 얼굴에 사각형 그리기
            roi_gray = gray[y:y+h, x:x+w] # 흑백 이미지에서 얼굴에 해당하는 부분
            roi_color = img[y:y+h, x:x+w] # 컬러 이미지에서 얼굴에 해당하는 부분
            eyes = eye_cascade.detectMultiScale(roi_gray) # 눈으로 인식된 좌표의 list
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2) # 원본 이미지의 눈에 녹색 사각형 그리기
        cv2.imwrite(path, img) # 원본 이미지에 사각형이 그려진 이미지 덮어쓰기

    else:
        print('Error occurred within cv_detect_face!')
        print(path)
