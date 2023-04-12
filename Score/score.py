# 선형 회귀 실습(최소제곱법)
# 학생별 공부 시간에 따른 성적 예측하기

import numpy as np
import matplotlib.pyplot as plt

# [공부한 시간, 성적] 데이터 값
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# 리스트로 되어 있는 x, y 값을 넘파이 배열로 바꾸기
# (인덱스를 주어 하나씩 불러와 계산이 가능하게 하기 위함)
x_data = np.array(x)
y_data = np.array(y)


# 경사 하강법 함수: MSE를 최소로 만드는 a, b 값 구하기
# 매개 변수는 (학습률, 학습 반복 횟수)
def gradient_descent(lr=0.03, epochs=2001):
    # 기울기 a와 y절편 b
    a, b = 0, 0

    for i in range(epochs):
        y_pred = a * x_data + b     # 예측값(y=ax+b) 구하기
        error = y_data - y_pred     # 오차(실제값과 예측값의 차이)

        # 오차 함수를 a로 미분한 값
        a_diff = -(2 / len(x_data)) * sum(x_data * (error))

        # 오차 함수를 b로 미분한 값
        b_diff = -(2 / len(x_data)) * sum(error)

        a = a - lr * a_diff
        b = b - lr * b_diff

        if i % 100 == 0:
            print(f"epoch={i}, 기울기={a:.04f}, 절편={b:.04f}")

    # 최종 기울기(a)와 절편(b) 리턴하기
    return a, b

a, b = gradient_descent(0.03, 2001)

# 그래프 출력
y_pred = a * x_data + b
plt.scatter(x, y)
plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)], color="red")
plt.show()

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