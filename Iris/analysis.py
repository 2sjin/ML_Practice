# 데이터 분석(전처리): 아이리스 꽃 품종 예측하기

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# csv 데이터 불러오기
header_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df = pd.read_csv("dataset_train.csv", names=header_names)

# 데이터의 첫 5줄과 끝 5줄 출력
print(df.head())
print("...")
print(df.tail())
print("-" * 50)

# 데이터 시각화: 상관도 그래프
sns.pairplot(df, hue='species')
plt.show()