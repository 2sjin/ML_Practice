from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

# 모델 불러오기
model = load_model("model.h5")

# 테스트 데이터셋 불러오기
df = pd.read_csv("dataset_test.csv")

# 데이터프레임을 2차원 넘파이 배열로 변환하고, 속성 데이터셋 추출
arr_2d = df.values                  # 2차원 넘파이 배열 형태로 변환
X = arr_2d[:, 0:4].astype(float)    # 속성 데이터셋

# 데이터셋에 모델 적용하여 예측 수행
y_predictions = model.predict(X)

# 가장 큰 값의 인덱스(품종) 반환
# 정수 0~2 사이의 1차원 배열이 됨
y_predictions = np.argmax(y_predictions, axis=-1)

# 예측 결과 출력
for i in range(len(y_predictions)):
    species = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")
    print(f"{i+1:03} - 예측값: {y_predictions[i]}, 실제값: {species.index(arr_2d[i, 4])}")