# 레드와인과 화이트와인 구분하기
# 모델 저장 조건 설정, 학습 자동 중단 설정

# 케라스 함수 및 라이브러리 호출
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf
import numpy as np
import pandas as pd

# 난수 시드 설정(시드가 같으면 실행할 때 마다 같은 결과 출력)
np.random.seed(5)
tf.random.set_seed(5)

# csv 데이터 불러오기
df = pd.read_csv("dataset_train.csv", header=None)

# 데이터를 X와 Y로 구분하기
data_set = df.values   # 2차원 넘파이 배열 형태로 변환
X = data_set[:, 0:12].astype(float)  # 속성 데이터셋
Y = data_set[:, 12]                  # 클래스 데이터셋(화이트와인=0, 레드와인=1)

# 딥러닝 모델 구성
model = Sequential()        # 모델 선언
model.add(Dense(30, input_dim=12, activation='relu'))   # 입력층 + 첫 번째 은닉층
model.add(Dense(12, activation='relu'))                 # 두 번째 은닉층
model.add(Dense(8, activation='relu'))                  # 세 번째 은닉층
model.add(Dense(1, activation='sigmoid'))               # 출력층

# 딥러닝 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 저장 조건 설정
# 테스트 오차가 최적(최소)일 때만 모델 저장
# verbose: 함수 진행사항 출력 여부(1이면 출력, 0이면 출력안함)
modelpath = "./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor="val_loss", verbose=1, save_best_only=True)

# 학습 자동 중단 설정
# 에포크 100회동안 테스트 오차가 좋아지지 않으면 학습 중단 
early_stooping_callback = EarlyStopping(monitor="val_loss", patience=100)

# 딥러닝 실행 및 모델 저장
# 전체 샘플을 {batch_size}개씩 끊어서 학습, {epochs}회 반복
# 각 에포크가 끝날 때마다 콜백 함수 호출됨
model.fit(X, Y, epochs=3500, batch_size=500, validation_split=0.2, verbose=0,
            callbacks=[early_stooping_callback, checkpointer])

# 정확도 출력
print(f"\n 정확도(Accuracy): {model.evaluate(X, Y)[1]:.4f}")
