# 선형 회귀 실습(최소제곱법)
# 학생별 공부 시간에 따른 성적 예측하기

import numpy as np

# y=ax+b에서, 기울기 a와 y절편 b
a, b = 3, 76

# [공부한 시간, 성적] 데이터 값
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# y=ax+b에 a와 b값을 대입하여 결과를 출력하는 함수(=예측값)
def predict(x):
    return a*x + b

# MSE(평균제곱오차) 함수: 작을수록 좋음
def mse(y, y_hat):
    return ((np.array(y) - np.array(y_hat)) ** 2).mean()

# 예측값 리스트
predict_list = []

# 예측값 리스트에 예측값 추가
for i in range(len(x)):
    predict_list.append(predict(x[i]))
    print(f"공부한 시간={x[i]}, 실제 점수={y[i]}, 예측 점수={predict(x[i])}")

# 최종 MES 출력
print("MSE =", mse(predict_list, y))