# '파마 인디언들의 당뇨병 여부' 예측하기

# 케라스 함수 및 라이브러리 호출
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np

# 난수 시드 설정(시드가 같으면 실행할 때 마다 같은 결과 출력)
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 불러오기
data_set = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# 피마 인디언의 정보와 당뇨병 발병 여부를 X와 Y로 구분하여 저장
X = data_set[:, 0:8]   # 속성 데이터셋
Y = data_set[:, 8]     # 클래스 데이터셋(당뇨병 발병 여부)

# 딥러닝 모델 선언
model = Sequential()

# 딥러닝 모델 구성
model.add(Dense(12, input_dim=8, activation='relu'))   # 입력층 + 첫 번째 은닉층
model.add(Dense(8, activation='relu'))          # 두 번째 은닉층
model.add(Dense(1, activation='sigmoid'))       # 출력층

# 딥러닝 모델 컴파일
# 오차 함수: 이항 교차 엔트로피
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 딥러닝 수행
# 전체 샘플을 {batch_size}개씩 끊어서 학습, {epochs}회 반복
model.fit(X, Y, epochs=100, batch_size=10)

# 정확도 출력
print(f"\n 정확도(Accuracy): {model.evaluate(X, Y)[1]:.4f}")