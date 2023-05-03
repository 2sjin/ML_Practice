import os
import shutil
from bing_image_downloader import downloader

import config


directory_list = [config.dir_train, config.dir_test]


# ==============================================================================================
# 필요한 라이브러리 설치하기
# ==============================================================================================
# git clone https://github.com/ndb796/bing_image_downloader


# ==============================================================================================
# 수집한 이미지를 학습 데이터와 평가 데이터로 구분하는 함수
# ==============================================================================================
def dataset_split(query, train_cnt):
    # 학습 및 평가 데이터셋 디렉토리 만들기
    for directory in directory_list:
        if not os.path.isdir(directory + '/' + query):
            os.makedirs(directory + '/' + query)
    # 학습 및 평가 데이터셋 준비하기
    cnt = 0
    for file_name in os.listdir(config.dir_rel + "/" + query):
        path_temp = config.dir_rel + '/' + query + '/' + file_name
        path_train = config.dir_train + '/' + query + '/' + file_name
        path_test = config.dir_test + '/' + query + '/' + file_name
        if cnt < train_cnt:
            print(f'[Train Dataset] {file_name}')
            shutil.move(path_temp, path_train)
        else:
            print(f'[Test Dataset] {file_name}')
            shutil.move(path_temp, path_test)
        cnt += 1
    shutil.rmtree(config.dir_rel + "/" + query)


# ==============================================================================================
# 초기 디렉토리 만들기
# ==============================================================================================
for directory in directory_list:
    if not os.path.isdir(directory):
        os.makedirs(directory)


# ==============================================================================================
# 이미지 크롤링 및 데이터셋 구축
# ==============================================================================================
for query in ['마동석', '김종국', '이병헌']:
    downloader.download(query, limit=40,  output_dir=config.dir_rel, adult_filter_off=True, force_replace=False, timeout=60)
    dataset_split(query, 30)