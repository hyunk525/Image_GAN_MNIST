import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

detector = dlib.get_frontal_face_detector() #얼굴을 찾아줌
shape = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat')

img = dlib.load_rgb_image('./imgs/02.jpg')
plt.figure(figsize=(16, 18))
plt.imshow(img)
plt.show()

#이미지 얼굴 위치에 네모칸 그리기
img_result = img.copy()  #10번 이미지를 복사 >> img_result는 복사된 이미지
dets = detector(img, 1)
fig, ax = plt.subplots(1, figsize=(10, 16))

for det in dets:
    x, y, w, h = det.left(), det.top(), det.width(), det.height()
    # 이미지에서 얼굴을 찾아서 x,y좌표+폭+높이 를 리스트로 잡음
    rect = patches.Rectangle((x, y), w, h, linewidth=4, edgecolor='y', facecolor='None')
    # facecolor >> 사각형 안의 색상 >> None줘야 사각형 테두리만 그려짐
    ax.add_patch(rect)
ax.imshow(img_result)
plt.show()

#랜드마크 찾기 - 양눈 양쪽 끝, 인중 >> 5개의 랜드마크에 포인트 찍기
# shape_predictor_68_face_landmarks >> 68개의 랜드마크에 포인트(얼굴윤곽, 눈썹 정보 등 더 많은 정보가짐)
# 얼굴 정렬을 맞춰줘야 하기 때문에 필요한 작업
fig, ax = plt.subplots(1, figsize=(16, 10))
obj = dlib.full_object_detections()

for detection in dets:
    s = shape(img, detection)
    obj.append(s)
    for point in s.parts():
        circle = patches.Circle((point.x, point.y), radius=3, edgecolor='b', facecolor='b')
        ax.add_patch(circle)
    ax.imshow(img_result)
plt.show()

