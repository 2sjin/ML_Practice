# 경사 하강법 함수: MSE를 최소로 만드는 a, b 값 구하기
# 매개 변수는 (학습률, 학습 반복 횟수)

def train(x, y, lr, epochs):
    # 기울기 a와 y절편 b
    a, b = 0, 0

    for i in range(epochs):
        y_pred = a * x + b     # 예측값(y=ax+b) 구하기
        error = y - y_pred     # 오차(실제값과 예측값의 차이)

        # 오차 함수를 a로 미분한 값
        a_diff = -(2 / len(x)) * sum(x * (error))

        # 오차 함수를 b로 미분한 값
        b_diff = -(2 / len(x)) * sum(error)

        # 학습률을 곱해 a, b 값 업데이트
        a = a - lr * a_diff
        b = b - lr * b_diff

        # epoch 100 단위로 데이터 출력
        if i % 100 == 0:
            print(f"epoch={i}, 기울기={a:.04f}, 절편={b:.04f}")

    # 최종 기울기(a)와 절편(b) 리턴하기
    return a, b