# 로지스틱 회귀 실습
# 학생별 공부 시간에 따른 합격/불합격 예측하기

# ============================================================

import logistic_gradient_descent

import numpy as np
import matplotlib.pyplot as plt

# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

# ============================================================

# [공부한 시간, 합격 여부] 데이터 값
data = [[2, 0], [4, 0], [6, 0], [8, 0], [10, 1], [12, 1], [14, 1]]

# 데이터를 넘파이 배열로 바꾸기
# (인덱스를 주어 하나씩 불러와 계산이 가능하게 하기 위함)
x = np.array([i[0] for i in data])
y = np.array([i[1] for i in data])

# 경사 하강법 함수: MSE를 최소로 만드는 a, b 값 구하기
# 매개 변수는 (학습률, 학습 반복 횟수)
a, b = logistic_gradient_descent.train(data, 0.05, 2001)

# 그래프 출력
plt.scatter(x, y)   # 실제 성적
plt.xlim(0, 15)
plt.ylim(-0.1, 1.1)
x_range = (np.arange(0, 15, 0.1))
plt.plot(np.arange(0, 15, 0.1), np.array([sigmoid(a*x + b) for x in x_range]), color="red")
plt.show()
