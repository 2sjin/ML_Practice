from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
import numpy as np
import pandas as pd

# 난수 시드 설정(시드가 같으면 실행할 때 마다 같은 결과 출력)
np.random.seed(3)
tf.random.set_seed(3)

# 모델 불러오기
model = load_model("model.h5")

# 테스트 데이터셋 불러오기
df = pd.read_csv("dataset_test.csv", header=None)
X = df[0].values.flatten()    # 리뷰 텍스트
Y = df[1].values.flatten()    # 긍정=1, 부정=0

# 텍스트 토큰화
token = Tokenizer()     # 토큰화 객체 생성
token.fit_on_texts(X)   # 토큰화 객체에 문장 적용
print(token.word_index) # 단어별 인덱스 값 출력

# 텍스트를 시퀀스 형태로 변환
seq_X = token.texts_to_sequences(X)

# 패딩: 서로 다른 길이의 데이터를 4로 통일함
padded_X = pad_sequences(seq_X, 4)

# 데이터셋에 모델 적용하여 예측 수행
Y_predictions = model.predict(padded_X).flatten()

# 예측 결과 출력
print("\n예측 결과")
for i in range(len(Y_predictions)):
    print(f"{X[i]} - 예측:{Y_predictions[i]:.4f}, 실제:{int(Y[i])}")
