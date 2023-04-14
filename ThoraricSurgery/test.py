from tensorflow.keras.models import load_model
import numpy as np

# 모델 불러오기
model = load_model("model.h5")

# 테스트 데이터셋 불러오기
test_data = np.loadtxt("dataset_test.csv", delimiter=",")

# 데이터셋에 모델 적용하여 예측 수행
predictions = model.predict(test_data[:, :17])

# 예측값 출력
for i in range(len(predictions)):
    print(f"{i+1:03}    예측값(성공확률): {predictions[i][0]:.4f}    실제값(수술결과): {int(test_data[i][17])}")