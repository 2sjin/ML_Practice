# 돌과 광물 구분하기
# k겹 교차 검증 실습

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder  # scikit-learn
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
import numpy as np
import pandas as pd

# 난수 시드 설정(시드가 같으면 실행할 때 마다 같은 결과 출력)
seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

# csv 데이터 불러오기
df = pd.read_csv("dataset.csv", header=None)

# 데이터를 X와 Y로 구분하기
dataset = df.values    # 2차원 넘파이 배열 형태로 변환
X = dataset[:, 0:60].astype(float)  # 속성 데이터셋
Y = dataset[:, 60]                  # 클래스 데이터셋

# 원-핫 인코딩: 클래스 데이터셋의 문자열 데이터를 숫자로 변환
e = LabelEncoder()
e.fit(Y)
Y = e.transform(Y)  # 클래스 데이터가 array([0, 1])로 바뀜

# 10겹 교차 검증 준비(데이터 10등분)
accuracy_list = []   # 10번의 테스트 정확도를 저장할 리스트
n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

# 10겹 교차 검증 실행
for train, test in skf.split(X, Y):
    # 딥러닝 모델 구성(시그모이드 모델)
    model = Sequential()        # 모델 선언
    model.add(Dense(24, input_dim=60, activation='relu'))   # 입력층 + 첫 번째 은닉층
    model.add(Dense(10, activation='relu'))                 # 두 번째 은닉층
    model.add(Dense(1, activation='sigmoid'))               # 출력층

    # 딥러닝 모델 컴파일
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 딥러닝 실행
    # 전체 샘플을 {batch_size}개씩 끊어서 학습, {epochs}회 반복
    model.fit(X[train], Y[train], epochs=100, batch_size=5)

    # 정확도 리스트에 저장
    current_accuracy = model.evaluate(X[test], Y[test])[1]
    accuracy_list.append(current_accuracy)

# 전체 정확도 출력
print(accuracy_list)