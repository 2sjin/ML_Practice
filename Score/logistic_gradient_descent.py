import numpy as np

# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

# 경사 하강법 함수: MSE를 최소로 만드는 a, b 값 구하기
# 매개 변수는 (학습률, 학습 반복 횟수)
def train(data, lr, epochs):
    # 기울기 a와 y절편 b
    a, b = 0, 0

    for i in range(epochs):
        for x, y in data:
            # a, b에 대한 편미분 값(sigmoid 함수 사용)
            a_diff = x * (sigmoid(a * x + b) - y)
            b_diff = sigmoid(a * x + b) - y

            # 학습률을 곱해 a, b 값 업데이트
            a = a - lr * a_diff
            b = b - lr * b_diff

            # epoch 1000 단위로 데이터 출력
            if i % 1000 == 0:
                print(f"epoch={i}, 기울기={a:.04f}, 절편={b:.04f}")

    # 최종 기울기(a)와 절편(b) 리턴하기
    return a, b