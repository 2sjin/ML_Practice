# 자연어 처리(NLP)
# 영화 리뷰의 긍정/부정 예측하기

from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
import numpy as np
import pandas as pd

# 난수 시드 설정(시드가 같으면 실행할 때 마다 같은 결과 출력)
np.random.seed(3)
tf.random.set_seed(3)

# csv 데이터 불러오기
df = pd.read_csv("dataset_train.csv", header=None, delimiter="|")
X = df[0].values.flatten()    # 리뷰 텍스트
Y = df[1].values.flatten()    # 긍정=1, 부정=0

# 텍스트 토큰화
token = Tokenizer()     # 토큰화 객체 생성
token.fit_on_texts(X)   # 토큰화 객체에 문장 적용
print(token.word_index) # 단어별 인덱스 값 출력

# 텍스트를 시퀀스 형태로 변환
seq_X = token.texts_to_sequences(X)
print("\n패딩 전\n", seq_X)

# 패딩: 서로 다른 길이의 데이터를 4로 통일함
# 길이가 짧으면 0으로 채우고, 길이가 짧으면 잘라서 맞춤
padded_X = pad_sequences(seq_X, 4)
print("\n패딩 후\n", padded_X)

# 임베딩에 입력될 단어 수 지정
word_size = len(token.word_index) + 1

# 딥러닝 모델 구성
model = Sequential()
model.add(Embedding(word_size, 8, input_length=4))  # 단어 임베딩(배열 압축)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# 딥러닝 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 딥러닝 실행 및 모델 저장
model.fit(padded_X, Y, epochs=20)

# 정확도 출력
print(f"\n 정확도(Accuracy): {model.evaluate(padded_X, Y)[1]:.4f}")

# 딥러닝 모델 저장
model.save("model.h5")