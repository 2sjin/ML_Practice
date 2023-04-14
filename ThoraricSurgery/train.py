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

# 딥러닝 모델 선언
model = Sequential()

# 입력층 + 은닉층
# 데이터에서 17개의 값을 받아 은닉층의 30개의 노드로 보냄
# 활성화 함수: 렐루 함수
model.add(Dense(30, input_dim=17, activation='relu'))

# 출력층(출력 노드 1개)
# 활성화 함수: 시그모이드 함수
model.add(Dense(1, activation='sigmoid'))

# 딥러닝 모델 컴파일
# 오차 함수: 이항 교차 엔트로피
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 딥러닝 수행
# 전체 샘플을 {batch_size}개씩 끊어서 학습, {epochs}회 반복
model.fit(X, Y, epochs=100, batch_size=10)