# 다중 선형 회귀 실습(최소제곱법)
# 학생별 공부 시간에 따른 성적 예측하기

# ============================================================

import linear_3d_gradient_descent

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# a와 b값을 대입하여 결과를 출력하는 함수(=예측값)
def predict(x1, x2):
    return a1*x1 + a2*x2 + b

# MSE(평균제곱오차) 함수: 작을수록 좋음
def mse(y, y_hat):
    return ((np.array(y) - np.array(y_hat)) ** 2).mean()

# ============================================================

# [공부한 시간(x1), 과외 수업 횟수(x2), 성적(y)] 데이터 값
data = [[2, 0, 81],
        [4, 4, 93],
        [6, 2, 91],
        [8, 3, 97]]

# 데이터를 넘파이 배열로 바꾸기
# (인덱스를 주어 하나씩 불러와 계산이 가능하게 하기 위함)
x1 = np.array([i[0] for i in data])
x2 = np.array([i[1] for i in data])
y = np.array([i[2] for i in data])

# 경사 하강법 함수: MSE를 최소로 만드는 a, b 값 구하기
# 매개 변수는 (x1, x2, y, 학습률, 학습 반복 횟수)
a1, a2, b = linear_3d_gradient_descent.train(x1, x2, y, 0.02, 2001)

# x1, x2로 3D 그래프에 출력하기 위한 예측값 계산
X1, X2 = np.meshgrid(x1, x2)
Y_pred = a1*X1 + a2*X2 + b

# 3D 그래프 출력
ax = plt.axes(projection="3d")
ax.set_xlabel("study_hours")
ax.set_ylabel("private_class")
ax.set_zlabel("Score")
ax.scatter(x1, x2, y)
ax.plot_surface(X1, X2, Y_pred, color="gray", alpha=0.3)
plt.show()

print()

# 예측값 리스트
predict_list = []

# 예측값 리스트에 예측값 추가
for i in range(len(x1)):
    predict_list.append(predict(x1[i], x2[i]))
    print(f"공부한 시간={x1[i]}, 과외 수업 횟수={x2[i]}, 실제 점수={y[i]}, 예측 점수={predict(x1[i], x2[i])}")

# 최종 MES 출력
print("\n평균제곱오차(MSE):", mse(predict_list, y), "\n")