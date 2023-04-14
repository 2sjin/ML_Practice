# '파마 인디언들의 당뇨병 여부' 데이터 분석 및 시각화
# (데이터 전처리 과정)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# csv 데이터 불러오고, 헤더 이름 지정
df = pd.read_csv("pima-indians-diabetes.csv",
                names = ["pregnant", "plasma", "pressure", "thickness",
                         "insulin", "BMI", "pedigree", "age", "class"])

# 데이터의 첫 5줄과 끝 5줄 출력
print(df.head(5))
print("...")
print(df.tail(5))
print("-" * 50)

# 데이터에 대한 정보 출력(행의 개수, 열의 이름과 타입 등)
print(df.info())
print("-" * 50)

# 데이터 요약 정보 출력
# (샘플 수, 평균, 표준편차, 최소값, 25%, 50%, 75%, 최대값)
print(df.describe())
print("-" * 50)

# 데이터 가공하기
# 임신 횟수(pregnant)별 당뇨병 발병(class) 평균 비율 구하기
print(df[['pregnant', 'class']]         # pregnant와 class 열만 출력
        .groupby(['pregnant'], as_index=False)          # pregnant 값에 따른 그룹화
        .mean()                                         # pregnant 그룹별 class의 평균값 구하기
        .sort_values(by='pregnant', ascending=True))    # pregnant를 기준으로 오름차순 정렬


# 데이터 시각화: 각 정보 간의 상관관계 구하기(heatmap)
# class 항목은 plasma(공복혈당) 항목과 가장 상관관계가 높음을 알 수 있음
plt.figure(figsize=(12, 12))
sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=plt.cm.gist_heat, linecolor="white", annot=True)
plt.show()

# 데이터 시각화: class별(0 or 1) plasma 값을 막대그래프로 표현
# 당뇨병 환자(class=1)의 공복혈당(plasma) 값이 100~200 사이인 경우가 많음
grid = sns.FacetGrid(df, col='class')
grid.map(plt.hist, 'plasma', bins=10)
plt.show()