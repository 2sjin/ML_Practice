# 선형 회귀 실습(최소제곱법)
# 학생별 공부 시간에 따른 성적 예측하기

# ============================================================

import linear_2d_gradient_descent

import numpy as np
import matplotlib.pyplot as plt

# y=ax+b에 a와 b값을 대입하여 결과를 출력하는 함수(=예측값)
def predict(x):
    return a*x + b

# MSE(평균제곱오차) 함수: 작을수록 좋음
def mse(y, y_hat):
    return ((np.array(y) - np.array(y_hat)) ** 2).mean()

# ============================================================

# [공부한 시간, 성적] 데이터 값
data = [[2, 81], [4, 93], [6, 91], [8, 97]]

# 데이터를 넘파이 배열로 바꾸기
# (인덱스를 주어 하나씩 불러와 계산이 가능하게 하기 위함)
x = np.array([i[0] for i in data])
y = np.array([i[1] for i in data])

# 경사 하강법 함수: MSE를 최소로 만드는 a, b 값 구하기
# 매개 변수는 (학습률, 학습 반복 횟수)
a, b = linear_2d_gradient_descent.train(x, y, 0.03, 2001)

# 그래프 출력
y_pred = predict(x)
plt.scatter(x, y)   # 실제 성적
plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color="red")     # 예측값(직선)
plt.show()

print()

# 예측값 리스트
predict_list = []

# 예측값 리스트에 예측값 추가
for i in range(len(x)):
    predict_list.append(predict(x[i]))
    print(f"공부한 시간={x[i]}, 실제 점수={y[i]}, 예측 점수={predict(x[i])}")

# 최종 MES 출력
print("\n평균제곱오차(MSE):", mse(predict_list, y), "\n")