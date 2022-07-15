import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

detector = dlib.get_frontal_face_detector() #얼굴을 찾아줌 >> 정면모습만!(frontal_face_detector)
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

#얼굴 정렬
def align_faces(img):
    dets = detector(img)
    objs = dlib.full_object_detections()
    for detection in dets:
        s = shape(img, detection)
        objs.append(s)
    faces = dlib.get_face_chips(img, objs, size=256, padding=0.35) #패딩>>이목구비 자른 사각형에 여유를 준것임
    return faces

#얼굴정렬 테스팅
test_img = dlib.load_rgb_image('./imgs/02.jpg')
test_faces = align_faces(test_img)
fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(20, 16))
axes[0].imshow(test_img)
for i, face in enumerate(test_faces):
    axes[i+1].imshow(face)
plt.show()

#화장 입히기
#텐서플로우
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

saver = tf.train.import_meta_graph('./models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')

#이미지 전처리(스케일링)
def preprocessing(img):
    return img / 127.5 - 1
#이미지 스케일링 한것(생성된 이미지) 복원
def deprocessing(img):
    return (img + 1) / 2

img1 = dlib.load_rgb_image('./imgs/14.jpg')
img1_faces = align_faces(img1)

img2 = dlib.load_rgb_image('./imgs/makeup/vFG56.png')
img2_faces = align_faces(img2)

#논메컵/메컵 사진 띄우기
fig, axes = plt.subplots(1, 2, figsize=(16, 10))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
plt.show()

src_img = img1_faces[0]
ref_img = img2_faces[0]

X_img = preprocessing(src_img)
X_img = np.expand_dims(X_img, axis=0)
Y_img = preprocessing(ref_img)
Y_img = np.expand_dims(Y_img, axis=0)

output = sess.run(Xs, feed_dict={X:X_img, Y:Y_img})  #X가 입력/Y가 출력
output_img = deprocessing(output[0])

fig, axes = plt.subplots(1, 3, figsize=(20, 10))
axes[0].imshow(src_img)
axes[1].imshow(ref_img)
axes[2].imshow(output_img)
plt.show()