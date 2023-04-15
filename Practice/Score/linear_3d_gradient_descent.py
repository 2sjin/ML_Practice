# 경사하강법(다중 선형 회귀)
# MSE를 최소로 만드는 a1, a2, b 값 구하기
# 매개 변수는 (x1, x2, y, 학습률, 학습 반복 횟수)

def train(x1, x2, y, lr, epochs):
    # 기울기 a1, a2와 y절편 b
    a1, a2, b = 0, 0, 0

    for i in range(epochs):
        y_pred = a1*x1 + a2*x2 + b     # 예측값(y=(a1*x1)+(a2*x2)+b) 구하기
        error = y - y_pred             # 오차(실제값과 예측값의 차이)

        # 오차 함수를 a1, a2, b로 미분한 값(순간기울기)
        a1_diff = -(2 / len(x1)) * sum(x1 * (error))
        a2_diff = -(2 / len(x2)) * sum(x2 * (error))
        b_diff = -(2 / len(x1)) * sum(error)

        # 학습률을 곱해 a, b 값 업데이트
        a1 = a1 - lr * a1_diff
        a2 = a2 - lr * a2_diff
        b = b - lr * b_diff

        # epoch 100 단위로 데이터 출력
        if i % 100 == 0:
            print(f"epoch={i}, 기울기1={a1:.04f}, 기울기2={a2:.04f}, 절편={b:.04f}")

    # 최종 기울기(a1, a2)와 절편(b) 리턴하기
    return a1, a2, b