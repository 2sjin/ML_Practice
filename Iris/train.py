# 다중 분류 문제
# 아이리스 꽃 품종 예측하기

# 케라스 함수 및 라이브러리 호출
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder  # scikit-learn

import tensorflow as tf
import numpy as np
import pandas as pd

# 난수 시드 설정(시드가 같으면 실행할 때 마다 같은 결과 출력)
np.random.seed(3)
tf.random.set_seed(3)

# csv 데이터 불러오기
header_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df = pd.read_csv("dataset_train.csv", names=header_names)

# 데이터를 X와 Y로 구분하기
data_set = df.values   # 2차원 넘파이 배열 형태로 변환
X = data_set[:, 0:4].astype(float)  # 속성 데이터셋
Y = data_set[:, 4]                  # 클래스 데이터셋(꽃의 품종)

# 원-핫 인코딩: 클래스 데이터셋(꽃의 품종) 문자열 데이터를 숫자로 변환
e = LabelEncoder()
e.fit(Y)
Y = e.transform(Y)                      # 품종이 array([1, 2, 3])로 바뀜
Y = tf.keras.utils.to_categorical(Y)    # 품종이 array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])로 바뀜

# 딥러닝 모델 구성(소프트맥스 모델)
model = Sequential()        # 모델 선언
model.add(Dense(16, input_dim=4, activation='relu'))    # 입력층 + 첫 번째 은닉층
model.add(Dense(3, activation='softmax'))               # 출력층

# 딥러닝 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 딥러닝 실행
# 전체 샘플을 {batch_size}개씩 끊어서 학습, {epochs}회 반복
model.fit(X, Y, epochs=50, batch_size=1)

# 정확도 출력
print(f"\n 정확도(Accuracy): {model.evaluate(X, Y)[1]:.4f}")

# 모델 저장하기
model.save("model.h5")