# 폐암 수술 환자의 생존율 예측하기 실습

# 케라스 함수 및 라이브러리 호출
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np

# 난수 시드 설정(시드가 같으면 실행할 때 마다 같은 결과 출력)
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 불러오기
data_set = np.loadtxt("ThoraricSurgery.csv", delimiter=",")

# 환자의 기록과 수술 결과를 X와 Y로 구분하여 저장
X = data_set[:, 0:17]   # 환자 기록(속성 데이터셋)
Y = data_set[:, 17]     # 수술 결과(클래스 데이터셋)

# 딥러닝 구조 결정(모델 설정) 및 실행: 2층 모델
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 딥러닝 실행
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=100, batch_size=10)