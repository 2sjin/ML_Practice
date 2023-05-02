# 필요한 라이브러리 설치하기
# git clone https://github.com/ndb796/bing_image_downloader

import os
import shutil
from bing_image_downloader import downloader

directory_list = [
    './CNN/custom_dataset/train/',
    './CNN/custom_dataset/test/',
]

# 초기 디렉토리 만들기
for directory in directory_list:
    if not os.path.isdir(directory):
        os.makedirs(directory)

# 수집한 이미지를 학습 데이터와 평가 데이터로 구분하는 함수
def dataset_split(query, train_cnt):
    # 학습 및 평가 데이터셋 디렉토리 만들기
    for directory in directory_list:
        if not os.path.isdir(directory + '/' + query):
            os.makedirs(directory + '/' + query)
    # 학습 및 평가 데이터셋 준비하기
    cnt = 0
    for file_name in os.listdir(query):
        if cnt < train_cnt:
            print(f'[Train Dataset] {file_name}')
            shutil.move(query + '/' + file_name, './CNN/custom_dataset/train/' + query + '/' + file_name)
        else:
            print(f'[Test Dataset] {file_name}')
            shutil.move(query + '/' + file_name, './CNN/custom_dataset/test/' + query + '/' + file_name)
        cnt += 1
    shutil.rmtree(query)

query = '마동석'
downloader.download(query, limit=40,  output_dir='./CNN/', adult_filter_off=True, force_replace=False, timeout=60)
dataset_split(query, 30)

query = '김종국'
downloader.download(query, limit=40,  output_dir='./CNN/', adult_filter_off=True, force_replace=False, timeout=60)
dataset_split(query, 30)

query = '이병헌'
downloader.download(query, limit=40,  output_dir='./CNN/', adult_filter_off=True, force_replace=False, timeout=60)
dataset_split(query, 30)