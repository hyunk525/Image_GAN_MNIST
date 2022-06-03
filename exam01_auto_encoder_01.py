import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist

#모델 만들기

#input_layer
input_img = Input(shape=(784,))
#encoding >> 압축시키기
encoded = Dense(32, activation='relu')
encoded = encoded(input_img)  #dense레이어는 입력이 있어야함 >> input_img / 출력=32
#decoding >> 다시 늘리기(복원)
decoded = Dense(784, activation='relu')
decoded = decoded(encoded)
autoencoder = Model(input_img, decoded)  #Model(입력값, 출력값)
autoencoder.summary()

encoder = Model(input_img, encoded)
encoder.summary()

encoded_input = Input(shape=(32,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
decoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

#데이터
(x_train, _), (x_test, _) = mnist.load_data() #y값 필요 없음 >> _ >> 분류기가 아님!
x_train = x_train / 255
x_test = x_test / 255
flatted_x_train = x_train.reshape(-1, 28*28)
flatted_x_test = x_test.reshape(-1, 28*28)
print(flatted_x_train.shape)  #(60000, 784)
print(flatted_x_test.shape)   #(10000, 784)

#학습 - encoder학습
fit_hist = autoencoder.fit(flatted_x_train, flatted_x_train, epochs=50, batch_size=256, validation_data=(flatted_x_test, flatted_x_test))  #입력된게 그대로 출력되어야 하기 때문

#encoded 이미지 따로 뽑아내기
encoded_img = encoder.predict(x_test[:10].reshape(-1, 28*28))

decoded_img = decoder.predict(encoded_img)

n = 10
plt.gray()
plt.figure(figsize=(20, 4))
for i in range(n):
    #2행 10열 첫번째줄
    ax = plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)  #x축 안보이게
    ax.get_yaxis().set_visible(False)  #y축 안보이게

    # 2행 10열 두번째줄
    ax = plt.subplot(2, 10, i+1+n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()