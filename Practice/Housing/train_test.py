# 집값 예측(선형 회귀)

# 케라스 함수 및 라이브러리 호출
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import tensorflow as tf
import numpy as np
import pandas as pd

# 난수 시드 설정(시드가 같으면 실행할 때 마다 같은 결과 출력)
seed = 0
np.random.seed(0)
tf.random.set_seed(3)

# csv 데이터 불러오기(공백을 구분자로 사용)
df = pd.read_csv("dataset.csv", delim_whitespace=True, header=None)

# 데이터를 X와 Y로 구분하기
data_set = df.values   # 2차원 넘파이 배열 형태로 변환
X = data_set[:, 0:13]  # 속성 데이터셋
Y = data_set[:, 13]    # 클래스 데이터셋(가격)

# 학습 데이터셋과 테스트 데이터셋 나누기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

# 딥러닝 모델 구성(선형 회귀)
model = Sequential()        # 모델 선언
model.add(Dense(30, input_dim=13, activation='relu'))   # 입력층 + 첫 번째 은닉층
model.add(Dense(6, activation='relu'))                  # 두 번째 은닉층
model.add(Dense(1))                                     # 출력층(활성화 함수 불필요)

# 딥러닝 모델 컴파일
model.compile(loss='mean_squared_error', optimizer='adam')

# 딥러닝 실행 및 모델 저장
# 전체 샘플을 {batch_size}개씩 끊어서 학습, {epochs}회 반복
model.fit(X_train, Y_train, epochs=200, batch_size=10)

# 딥러닝 모델 저장
model.save("model.h5")

# 메모리 내의 모델 삭제 및 새로 불러오기
del model
model = load_model("model.h5")

# 테스트(예측 값과 실제 값의 비교)
Y_prediction = model.predict(X_test).flatten()  # 예측값을 1차원 넘파이 배열로 만들기
for i in range(20):
    print(f"실제가격:{Y_test[i]:.3f}, 예상가격:{Y_prediction[i]:.3f}")