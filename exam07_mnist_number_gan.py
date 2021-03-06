# -*- coding: utf-8 -*-
"""exam07_MNIST_Number_Gan.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1x4Fk2bSEPP86olJN-Zeswx3_FgyNN6ZF
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist
import os

OUT_DIR = './OUT_img'
img_shape = (28, 28, 1)
epoch = 100000      #아래의 학습을 10만번 하기
batch_size = 128    #한 에폭당 랜덤한 128개의 이미지 학습
noise = 100
sample_interval = 100     #100에폭당 한번씩 이미지 생성을 통해 학습 확인

row = col = 4

(X_train, Y_train), (_, _) = mnist.load_data()
#라벨이랑 테스트 필요 없음
print(X_train.shape)

MY_NUM = 8
X_train = X_train[Y_train == MY_NUM]

_, axs = plt.subplots(row, col, figsize=(row, col), sharey=True, sharex=True)
        
cnt=0
for i in range(row):
    for j in range(col):
        axs[i, j].imshow(X_train[cnt, :, :], cmap='gray')
        axs[i, j].axis('off')
        cnt += 1
plt.show()

X_train = X_train / 127.5 -1   #x_train 픽셀값([0, 255])을 -1~1 값으로 스케일링
X_train = np.expand_dims(X_train, axis=3)  # X_train = X_train.reshape(-1, 28, 28, 1) 랑 같은 거임!
print(X_train.shape)

#이진분류기 >> 라벨만들기

real = np.ones((batch_size, 1))  # >> discriminator_model 실제 라벨 줄때 사용(1)
fake = np.zeros((batch_size, 1))

#GAN 모델 만들기

generator_model = Sequential()
generator_model.add(Dense(256*7*7, input_dim=noise))
generator_model.add(Reshape((7, 7, 256)))
generator_model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))  # trides >> 커널사이즈 옆으로 두칸씩 이동 >> 패딩same 줘도 같지 않음
generator_model.add(BatchNormalization())  #batch별로 중간중간 노멀라이제이션 >> 발산 방지
generator_model.add(LeakyReLU(alpha=0.01))
generator_model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
generator_model.add(BatchNormalization())
generator_model.add(LeakyReLU(alpha=0.01))
generator_model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
generator_model.add(Activation('tanh'))
generator_model.summary()

Irelu = LeakyReLU(alpha=0.01)
discriminator_model = Sequential()
discriminator_model.add(Flatten(input_shape=img_shape))
discriminator_model.add(Dense(128, activation=Irelu))
discriminator_model.add(Dense(1, activation='sigmoid'))
discriminator_model.summary()

discriminator_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#학습 시켜야하기 때문에 compile 시킴(레이어 학습)

#제네레이터 모델을 디스크리미네이터 학습이 필요함
#생성자와 판별자를 하나로 합쳐서 gan 모델을 만듦

gan_model = Sequential()
gan_model.add(generator_model)
gan_model.add(discriminator_model)
discriminator_model.trainable=False  #discriminator_model 는 학습할 수 없는 상태(모델학습) / 레이어 학습은 가능
gan_model.summary()

gan_model.compile(loss='binary_crossentropy', optimizer='adam')

#discriminator 학습 시키기 >> 이미지 생성을 해야함
for itr in range(epoch):
    idx = np.random.randint(0, X_train.shape[0], batch_size)  #shape[0] = 6만
    real_img = X_train[idx]

    #생성하기 >> 잡음을 만들어야함
    z = np.random.normal(0, 1, (batch_size, noise))
    fake_img = generator_model.predict(z)
    #잡음으로 가짜 이미지 생성

    #discriminator 학습
    d_hist_real = discriminator_model.train_on_batch(real_img, real)  #train_on_batch >> 한 batch 학습하고 끝남
    d_hist_fake = discriminator_model.train_on_batch(fake_img, fake)

    #loss값 평균 저장
    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake)

    #gan model 학습 >> generator만 학습됨 >> 위에서 discriminator trainale=False 줌
    z = np.random.normal(0, 1, (batch_size, noise))  #잡음만들기
    gan_hist = gan_model.train_on_batch(z, real)

    if itr % sample_interval == 0 :
        print('%d [D loss : %f, acc. : %0.2f%%] [G loss : %f]'%(itr, d_loss, d_acc*100, gan_hist))
        row = col = 4
        z = np.random.normal(0, 1, (row * col, noise))
        fake_img = generator_model.predict(z)
        fake_img = 0.5 * fake_img + 0.5
        _, axs = plt.subplots(row, col, figsize=(row, col), sharey=True, sharex=True)
        
        cnt=0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(fake_img[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        plt.show()
        generator_model.save('./generator_mnist_{}_hk.h5'.format(MY_NUM))